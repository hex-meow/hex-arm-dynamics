//! 机械臂动力学(从 URDF):正运动学 + 重力力矩 `G(q)`。
//!
//! 仅依赖 `urdf-rs` 解析;线代手写(无 nalgebra)。重力力矩供两处**同一模型**:
//!   - 控制器 GRAVITY_COMP 前馈 `tau_ff = G(q)`;
//!   - 仿真 plant 的重力负载(EoM 里减去 `G(q)`,臂在重力下会下垂)。
//! 二者用同一 `G` → 完美抵消(测通路/可视化);`G` 的**物理正确性**由单元测试(单摆解析解)
//! 与真机最终验证。EoM 约定:`M q̈ + C q̇ + G(q) = τ`,静止保持 `τ = G(q)`。
//!
//! 假设:base→tip 串联、全 revolute(firefly_y6 即是);base_link 竖直安装于世界原点。

use std::collections::HashMap;

use anyhow::{anyhow, Result};

type V3 = [f32; 3];
type M3 = [[f32; 3]; 3]; // 行优先

const G_ACC: f32 = 9.81;

// ── 极简线代 ──
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: V3, b: V3) -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn sub(a: V3, b: V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn add(a: V3, b: V3) -> V3 { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn scale(a: V3, s: f32) -> V3 { [a[0] * s, a[1] * s, a[2] * s] }
fn matvec(m: &M3, v: V3) -> V3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}
fn matmul(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}
fn identity() -> M3 { [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] }

/// URDF rpy(固定轴 XYZ):R = Rz(yaw)·Ry(pitch)·Rx(roll)。
fn rpy_to_mat(r: f32, p: f32, y: f32) -> M3 {
    let (cr, sr) = (r.cos(), r.sin());
    let (cp, sp) = (p.cos(), p.sin());
    let (cy, sy) = (y.cos(), y.sin());
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

/// Rodrigues:绕单位轴 `axis` 转 `ang`。
fn axis_angle_to_mat(axis: V3, ang: f32) -> M3 {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if n < 1e-9 { return identity(); }
    let k = [axis[0] / n, axis[1] / n, axis[2] / n];
    let (s, c) = (ang.sin(), ang.cos());
    let v = 1.0 - c;
    [
        [c + k[0] * k[0] * v, k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s],
        [k[1] * k[0] * v + k[2] * s, c + k[1] * k[1] * v, k[1] * k[2] * v - k[0] * s],
        [k[2] * k[0] * v - k[1] * s, k[2] * k[1] * v + k[0] * s, c + k[2] * k[2] * v],
    ]
}

#[derive(Clone)]
struct JointDef {
    origin_xyz: V3,
    origin_rot: M3,
    axis: V3,
}
#[derive(Clone)]
struct LinkDef {
    mass: f32,
    com: V3, // 在该 link 坐标系
    mesh: Option<String>,
}

/// URDF 片段(EE/工具)的**集总惯量**:零位形下沿 joint origin 变换遍历整棵树(任意
/// joint 类型、支持分叉),把所有 link 的 (mass, com) 汇总为「根 link 坐标系」下的
/// (总质量, 总 COM)。用途:把 EE 折叠为臂 tip 的固定载荷(gravity comp)——抓手自身
/// 关节运动带来的 COM 偏移在此近似下忽略。无 inertial 的 link 记 0 质量;总质量为 0
/// 时返回 (0, [0;3])(调用侧自行告警)。
pub fn lumped_inertial_from_urdf_string(xml: &str) -> Result<(f32, [f32; 3])> {
    let robot = urdf_rs::read_from_string(xml).map_err(|e| anyhow!("解析 URDF 字串: {e}"))?;
    let mv = |m: &M3, v: &V3| -> V3 {
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    };
    let mm = |a: &M3, b: &M3| -> M3 {
        let mut r = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        r
    };
    // link 名 → (mass, com)
    let mut info: std::collections::HashMap<&str, (f32, V3)> = std::collections::HashMap::new();
    for l in &robot.links {
        let (mass, com) = if l.inertial.mass.value > 0.0 {
            (
                l.inertial.mass.value as f32,
                [l.inertial.origin.xyz[0] as f32, l.inertial.origin.xyz[1] as f32, l.inertial.origin.xyz[2] as f32],
            )
        } else {
            (0.0, [0.0; 3])
        };
        info.insert(l.name.as_str(), (mass, com));
    }
    // 根 = 从不作为 child 的 link
    let children: std::collections::HashSet<&str> = robot.joints.iter().map(|j| j.child.link.as_str()).collect();
    let root = robot
        .links
        .iter()
        .map(|l| l.name.as_str())
        .find(|n| !children.contains(n))
        .ok_or_else(|| anyhow!("URDF 片段找不到根 link"))?;
    // BFS:accumulate (rot, pos) —— 零位形,只用 joint origin
    let mut total_m = 0.0f32;
    let mut moment = [0.0f32; 3];
    let mut queue: Vec<(&str, M3, V3)> = vec![(root, identity(), [0.0; 3])];
    while let Some((name, rot, pos)) = queue.pop() {
        if let Some((m, com)) = info.get(name) {
            if *m > 0.0 {
                let world = {
                    let rc = mv(&rot, com);
                    [pos[0] + rc[0], pos[1] + rc[1], pos[2] + rc[2]]
                };
                total_m += m;
                for k in 0..3 {
                    moment[k] += m * world[k];
                }
            }
        }
        for j in robot.joints.iter().filter(|j| j.parent.link == name) {
            let jrot = rpy_to_mat(j.origin.rpy[0] as f32, j.origin.rpy[1] as f32, j.origin.rpy[2] as f32);
            let jxyz = [j.origin.xyz[0] as f32, j.origin.xyz[1] as f32, j.origin.xyz[2] as f32];
            let step = mv(&rot, &jxyz);
            queue.push((
                j.child.link.as_str(),
                mm(&rot, &jrot),
                [pos[0] + step[0], pos[1] + step[1], pos[2] + step[2]],
            ));
        }
    }
    if total_m <= 0.0 {
        return Ok((0.0, [0.0; 3]));
    }
    Ok((total_m, [moment[0] / total_m, moment[1] / total_m, moment[2] / total_m]))
}

/// 串联臂的动力学/运动学模型。
#[derive(Clone)]
pub struct ArmDynamics {
    joints: Vec<JointDef>,
    links: Vec<LinkDef>,
    gravity: V3, // 世界系重力向量(默认 [0,0,-9.81])
}

/// 一个 link 的世界位姿(旋转 + 平移),供可视化 / FK 消费。
#[derive(Debug, Clone, Copy)]
pub struct LinkPose {
    pub rot: M3,
    pub pos: V3,
    pub mesh_idx: usize,
}

impl ArmDynamics {
    pub fn from_urdf_file(path: &str) -> Result<Self> {
        let robot = urdf_rs::read_file(path).map_err(|e| anyhow!("读 URDF {path}: {e}"))?;
        Self::from_robot(robot)
    }

    /// 从 URDF XML 字串解析(GUI/host 从 `arm/urdf` queryable 取到 xml 后建模型用)。
    pub fn from_urdf_string(xml: &str) -> Result<Self> {
        let robot = urdf_rs::read_from_string(xml).map_err(|e| anyhow!("解析 URDF 字串: {e}"))?;
        Self::from_robot(robot)
    }

    fn from_robot(robot: urdf_rs::Robot) -> Result<Self> {
        // link 名 → (mass, com, mesh)
        let mut link_info: HashMap<String, LinkDef> = HashMap::new();
        for l in &robot.links {
            let (mass, com) = match &l.inertial {
                i if i.mass.value > 0.0 => (
                    i.mass.value as f32,
                    [i.origin.xyz[0] as f32, i.origin.xyz[1] as f32, i.origin.xyz[2] as f32],
                ),
                _ => (0.0, [0.0; 3]),
            };
            let mesh = l.visual.first().and_then(|v| match &v.geometry {
                urdf_rs::Geometry::Mesh { filename, .. } => Some(filename.clone()),
                _ => None,
            });
            link_info.insert(l.name.clone(), LinkDef { mass, com, mesh });
        }

        // 找根 link(从不作为某 joint 的 child)
        let children: std::collections::HashSet<&str> =
            robot.joints.iter().map(|j| j.child.link.as_str()).collect();
        let root = robot
            .links
            .iter()
            .map(|l| l.name.as_str())
            .find(|n| !children.contains(n))
            .ok_or_else(|| anyhow!("URDF 找不到根 link"))?;

        // 从根沿 parent→child 串起来
        let joint_by_parent: HashMap<&str, &urdf_rs::Joint> =
            robot.joints.iter().map(|j| (j.parent.link.as_str(), j)).collect();

        let mut joints = Vec::new();
        let mut links = Vec::new();
        let mut cur = root;
        while let Some(j) = joint_by_parent.get(cur) {
            match &j.joint_type {
                urdf_rs::JointType::Revolute | urdf_rs::JointType::Continuous => {}
                other => return Err(anyhow!("joint {} 类型 {:?} 暂不支持(只支持 revolute 串联)", j.name, other)),
            }
            joints.push(JointDef {
                origin_xyz: [j.origin.xyz[0] as f32, j.origin.xyz[1] as f32, j.origin.xyz[2] as f32],
                origin_rot: rpy_to_mat(j.origin.rpy[0] as f32, j.origin.rpy[1] as f32, j.origin.rpy[2] as f32),
                axis: [j.axis.xyz[0] as f32, j.axis.xyz[1] as f32, j.axis.xyz[2] as f32],
            });
            let child = j.child.link.as_str();
            links.push(link_info.remove(child).ok_or_else(|| anyhow!("link {child} 缺 inertial"))?);
            cur = child;
        }
        if joints.is_empty() {
            return Err(anyhow!("URDF 没解析出任何 revolute 关节"));
        }
        Ok(Self { joints, links, gravity: [0.0, 0.0, -G_ACC] })
    }

    pub fn dof(&self) -> usize { self.joints.len() }

    /// 直接构造(测试用)。
    pub fn from_parts(joints_data: Vec<(V3, M3, V3)>, links_data: Vec<(f32, V3)>, gravity: V3) -> Self {
        let joints = joints_data.into_iter().map(|(origin_xyz, origin_rot, axis)| JointDef { origin_xyz, origin_rot, axis }).collect();
        let links = links_data.into_iter().map(|(mass, com)| LinkDef { mass, com, mesh: None }).collect();
        Self { joints, links, gravity }
    }

    /// FK 内核:返回每个关节的(世界轴, 世界原点)与每个 link 的(世界旋转, 世界原点, 世界质心)。
    fn forward(&self, q: &[f32]) -> (Vec<(V3, V3)>, Vec<(M3, V3, V3)>) {
        let mut r_cum = identity();
        let mut p_cum = [0.0f32; 3];
        let mut joint_world = Vec::with_capacity(self.dof());
        let mut link_world = Vec::with_capacity(self.dof());
        for i in 0..self.dof() {
            let j = &self.joints[i];
            let p_j = add(p_cum, matvec(&r_cum, j.origin_xyz)); // 关节原点(世界)
            let r_after_origin = matmul(&r_cum, &j.origin_rot);
            let axis_world = matvec(&r_after_origin, j.axis);
            joint_world.push((axis_world, p_j));

            let r_axis = axis_angle_to_mat(j.axis, q[i]);
            let r_child = matmul(&r_after_origin, &r_axis);
            let p_child = p_j; // 绕过原点的轴旋转不平移
            let com_world = add(p_child, matvec(&r_child, self.links[i].com));
            link_world.push((r_child, p_child, com_world));

            r_cum = r_child;
            p_cum = p_child;
        }
        (joint_world, link_world)
    }

    /// 重力力矩 `G(q)`,用模型自带的重力向量(默认 [0,0,-9.81])。
    /// 把固定 tip 载荷(如 EE 集总惯量)折叠进**最后一个 link**:质量相加、COM 按质量
    /// 加权合并。`com` 在最后一个 link 坐标系表达(EE 以单位变换 fixed joint 挂 tip 时,
    /// 即 EE 挂载根系 == tip link 系)。返回新模型,原模型不变(便于换 EE 时从原模型重折)。
    pub fn with_tip_payload(&self, mass: f32, com: [f32; 3]) -> Self {
        let mut m = self.clone();
        if mass > 0.0 {
            if let Some(last) = m.links.last_mut() {
                let m0 = last.mass;
                let mt = m0 + mass;
                for k in 0..3 {
                    last.com[k] = (m0 * last.com[k] + mass * com[k]) / mt;
                }
                last.mass = mt;
            }
        }
        m
    }

    pub fn gravity_torque(&self, q: &[f32]) -> Vec<f32> {
        self.gravity_torque_with(q, self.gravity)
    }

    /// 重力力矩 `G(q)`,用**指定** base 系重力向量(m/s²)。斜装/重力计驱动用。
    /// `G_j = -Σ_{i≥j} ẑ_j · ((p_ci − p_j) × (m_i·g))`,静止保持 `τ = G(q)`。
    pub fn gravity_torque_with(&self, q: &[f32], gravity: [f32; 3]) -> Vec<f32> {
        let (joint_world, link_world) = self.forward(q);
        let mut g = vec![0.0f32; self.dof()];
        for (j, (axis_j, p_j)) in joint_world.iter().enumerate() {
            let mut tau = 0.0f32;
            for i in j..self.dof() {
                let (_, _, com_i) = link_world[i];
                let weight = scale(gravity, self.links[i].mass); // m_i·g(指向下)
                tau += dot(*axis_j, cross(sub(com_i, *p_j), weight));
            }
            g[j] = -tau;
        }
        g
    }

    /// 各 link 世界位姿(供 rerun 可视化)。
    pub fn link_poses(&self, q: &[f32]) -> Vec<LinkPose> {
        let (_, link_world) = self.forward(q);
        link_world.iter().enumerate().map(|(i, (rot, pos, _))| LinkPose { rot: *rot, pos: *pos, mesh_idx: i }).collect()
    }

    /// 各 link 的网格文件名(package:// 形式,可能为 None)。
    pub fn link_meshes(&self) -> Vec<Option<String>> {
        self.links.iter().map(|l| l.mesh.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 单摆:关节轴 +y 在原点,连杆质量 m,质心在 +x 距离 l。
    /// q=0(水平)时 G = -m·g·l;q=π/2(竖直向下,质心在 -z)时 G≈0。
    #[test]
    fn pendulum_gravity() {
        let m = 2.0f32;
        let l = 0.5f32;
        let dyn_ = ArmDynamics::from_parts(
            vec![([0.0, 0.0, 0.0], identity(), [0.0, 1.0, 0.0])],
            vec![(m, [l, 0.0, 0.0])],
            [0.0, 0.0, -G_ACC],
        );
        let g0 = dyn_.gravity_torque(&[0.0]);
        assert!((g0[0] - (-m * G_ACC * l)).abs() < 1e-3, "G(0)={} 期望 {}", g0[0], -m * G_ACC * l);

        // 转到质心朝正下方(绕 +y 转 +90° 把 +x 转到 -z)→ 力臂为零
        let g90 = dyn_.gravity_torque(&[std::f32::consts::FRAC_PI_2]);
        assert!(g90[0].abs() < 1e-2, "G(π/2)={} 期望 ≈0", g90[0]);
    }

    /// 二连杆:验证下游连杆对上游关节的贡献叠加。
    #[test]
    fn two_link_sum() {
        // 两个 +y 轴关节,连杆各在 +x,joint2 origin 在 [l,0,0]
        let l = 0.4f32;
        let dyn_ = ArmDynamics::from_parts(
            vec![
                ([0.0, 0.0, 0.0], identity(), [0.0, 1.0, 0.0]),
                ([l, 0.0, 0.0], identity(), [0.0, 1.0, 0.0]),
            ],
            vec![(1.0, [l, 0.0, 0.0]), (1.0, [l, 0.0, 0.0])],
            [0.0, 0.0, -G_ACC],
        );
        let g = dyn_.gravity_torque(&[0.0, 0.0]);
        // joint2 只托 link2:质心在 x=2l,力臂 l → |G2| = m·g·l
        assert!((g[1].abs() - 1.0 * G_ACC * l).abs() < 1e-2);
        // joint1 托两根:link1 质心 x=l,link2 质心 x=2l → |G1| = g·(l + 2l) = 3 g l
        assert!((g[0].abs() - G_ACC * 3.0 * l).abs() < 1e-2);
    }
}

#[cfg(test)]
mod payload_tests {
    use super::*;

    const EE: &str = r#"<robot name="ee">
      <link name="ee_base"><inertial><origin xyz="0 0 0.05"/><mass value="0.4"/><inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="finger"><inertial><origin xyz="0 0 0.01"/><mass value="0.1"/><inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <joint name="j" type="revolute"><origin xyz="0 0 0.1" rpy="0 0 0"/><parent link="ee_base"/><child link="finger"/><axis xyz="0 0 1"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>
    </robot>"#;

    #[test]
    fn lumped_inertial_tree_at_zero_pose() {
        let (m, com) = lumped_inertial_from_urdf_string(EE).unwrap();
        // 总质量 0.5;COM_z = (0.4*0.05 + 0.1*(0.1+0.01))/0.5 = 0.062
        assert!((m - 0.5).abs() < 1e-6);
        assert!((com[2] - 0.062).abs() < 1e-6, "com={com:?}");
        assert!(com[0].abs() < 1e-6 && com[1].abs() < 1e-6);
    }

    #[test]
    fn tip_payload_folds_into_gravity_torque() {
        // 单关节水平臂:关节在原点绕 y,link COM 在 x=1、质量 1kg → G = -(-9.81) 视符号约定
        let arm = ArmDynamics::from_parts(
            vec![([0.0, 0.0, 0.0], identity(), [0.0, 1.0, 0.0])],
            vec![(1.0, [1.0, 0.0, 0.0])],
            [0.0, 0.0, -9.81],
        );
        let g0 = arm.gravity_torque(&[0.0])[0];
        // 折叠 1kg 载荷在同一 COM → 力矩恰好翻倍
        let g1 = arm.with_tip_payload(1.0, [1.0, 0.0, 0.0]).gravity_torque(&[0.0])[0];
        assert!((g1 - 2.0 * g0).abs() < 1e-4, "g0={g0} g1={g1}");
        // 原模型不受影响
        assert!((arm.gravity_torque(&[0.0])[0] - g0).abs() < 1e-9);
    }
}
