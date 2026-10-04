"""Real-physics foraging checks and isolated ingestion/gate regression cases."""
import copy
import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from .foraging import ForageConfig,ForagingEnv,apply_food_profile
from .forage_learning import promotion_decision


def run_checks():
    short=ForagingEnv(ForageConfig(episode_seconds=0.2,food_max=2))
    try:
        check_env(short,skip_render_check=True)
        a,_=short.reset(seed=77)
        positions=short.food_positions.copy()
        b,_=short.reset(seed=77)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(positions,short.food_positions)
        assert np.all(short.frame[48:50]==0),"Pure sugar must not create volatile odor"
        # A far-away food source must not be eaten even with a feeding command.
        short.reset(seed=1,options={"food_positions":[[20,20]]})
        for _ in range(20):
            amount,feeding=short._ingest(1.0,0.0)
            assert amount==0 and not feeding
        # Isolate mouth contact accounting at the actual current mouth position.
        short.food_positions[0,:2]=short.mouth_position[:2]
        short.sim.mj_data.site_xpos[short.mouth_site,2]=0.12
        assert short._ingest(-1.0,0.0)==(0.0,False)
        assert short._ingest(1.0,10.0)==(0.0,False)
        for _ in range(100):
            short._ingest(1.0,0.0)
        assert np.isclose(short.consumed,1.0)
        assert short.food_remaining[0]==0
        assert short._ingest(1.0,0.0)==(0.0,False),"Consumed food must not pay twice"
        short.reset(seed=77)
        assert np.all(short.food_remaining[:short.food_count]==1)
        options=mujoco.MjvOption()
        for geom in short.food_geoms[:short.food_count]:
            assert options.geomgroup[short.sim.mj_model.geom_group[geom]],"Food must be visible by default"
            assert short.sim.mj_model.geom_rgba[geom,3]==1,"Reset must restore depleted food"
        print("PASS: reproducible sensors/layout, no sugar odor, proximity/mouth/speed gates, mass accounting",flush=True)
    finally:
        short.close()
    env=ForagingEnv(ForageConfig(episode_seconds=4,food_min=1,food_max=1))
    try:
        env.reset(seed=0,options={"food_positions":[[6,0]]})
        for _ in range(env.max_steps):
            _,_,terminated,truncated,info=env.step(np.zeros(3))
            if terminated or truncated:
                break
        assert info["metrics"]["ate_all"] and not info["metrics"]["fallen"]
        assert np.isclose(sum(event["amount"] for event in env.events),1.0)
        assert env.first_ingestion_time>0.1
        for event in env.events:
            assert np.linalg.norm(np.array(event["mouth_xyz_mm"][:2])-[6,0])<=0.85
        print("PASS: actual fly approaches, halts, extends proboscis and consumes a reachable sugar patch",flush=True)
    finally:
        env.close()
    config=apply_food_profile(ForageConfig(episode_seconds=4,food_min=1,food_max=1),"sucrose-droplet")
    droplet=ForagingEnv(config)
    try:
        droplet.reset(seed=0,options={"food_positions":[[6,0]]})
        geom=droplet.food_geoms[0]
        assert droplet.sim.mj_model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_ELLIPSOID
        axes=droplet.sim.mj_model.geom_size[geom]
        assert np.isclose(4*np.pi*np.prod(axes)/3,0.1),"Visible volume must be 0.1 microlitres"
        assert np.isclose(2*axes[0],0.8),"Droplet diameter must be 0.8 mm"
        center=droplet.food_positions[0].copy()
        assert droplet._near(center,0.05,0.25)[0]
        assert not droplet._near(center+[0,0,axes[2]+0.06],0.05,0.25)[0],"No feeding above the visible drop"
        assert not droplet._near(center+[axes[0]+0.06,0,0],0.05,0.25)[0],"No feeding beyond the drop edge"
        for _ in range(droplet.max_steps):
            _,_,terminated,truncated,info=droplet.step(np.zeros(3))
            if terminated or truncated:
                break
        assert info["metrics"]["ate_any"] and not info["metrics"]["fallen"]
        for event in droplet.events:
            assert np.sum(((np.asarray(event["mouth_xyz_mm"])-center)/(axes+0.05))**2)<=1+1e-8
        assert np.isclose(droplet.consumed+droplet.food_remaining.sum(),1.0)
        print("PASS: 0.8-mm/0.1-uL droplet geometry, tighter contact and physical ingestion",flush=True)
    finally:
        droplet.close()
    baseline={"rows":[{"seed":i,"consumed_fraction":0.2} for i in range(8)],
              "fall_rate":0.0,"escape_rate":0.0,"biology":{"available":False}}
    assert not promotion_decision(baseline,baseline)["promote"]
    better=copy.deepcopy(baseline)
    for row in better["rows"]:
        row["consumed_fraction"]=0.6
    verdict=promotion_decision(baseline,better)
    assert verdict["promote"] and not verdict["biological_improvement_supported_on_reference"]
    better["fall_rate"]=0.1
    assert not promotion_decision(baseline,better)["promote"]
    better["fall_rate"]=0.0
    better["rows"][0]["consumed_fraction"]=0.1
    assert not promotion_decision(baseline,better)["promote"],"Mean improvement cannot hide a regressed paired trial"
    biological=copy.deepcopy(baseline)
    biological["biology"]={"available":True,"distance":1.0,"reference_sha256":"synthetic-gate-test"}
    assert not promotion_decision(biological,biological)["promote"]
    closer=copy.deepcopy(biological)
    closer["biology"]["distance"]=0.9
    assert promotion_decision(biological,closer)["promote"]
    closer["biology"]["reference_sha256"]="different-test-reference"
    try:
        promotion_decision(biological,closer)
    except ValueError:
        pass
    else:
        raise AssertionError("A changed biological reference must invalidate comparison")
    print("PASS: reject flat/regressed candidates; task improvement is not labeled biological similarity",flush=True)
