import time

import mujoco
import mujoco.viewer

XML = """
<mujoco model="cyberfly_setup_check">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 -2 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1"
          rgba="0.3 0.35 0.4 1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.08" mass="0.02"
            rgba="0.9 0.4 0.15 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    with viewer.lock():
        viewer.cam.lookat[:] = [0, 0, 0.4]
        viewer.cam.distance = 3
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

    while viewer.is_running():
        start = time.perf_counter()

        # 每两秒重置，让小球反复下落。
        if data.time >= 2:
            mujoco.mj_resetData(model, data)

        mujoco.mj_step(model, data)
        viewer.sync()

        remaining = model.opt.timestep - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)
