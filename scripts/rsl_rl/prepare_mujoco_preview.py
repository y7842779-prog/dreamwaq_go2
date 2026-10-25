"""Export the three final checkpoints into isolated, keyboard-controlled C++ bundles.

Uses existing Unitree binaries and observation headers without changing either
external repository. Validation is CPU-only; it never opens a DDS connection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = {"Asym_PPO": "2026-09-09_21-44-06", "DreamWaQ_CENet": "2026-09-11_20-51-03", "PayloadEstimator": "2026-10-16_21-12-27"}
TRAIN_JOINTS = [f"{leg}_{kind}_joint" for kind in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]
SDK_JOINTS = [f"{leg}_{kind}_joint" for leg in ("FR", "FL", "RR", "RL") for kind in ("hip", "thigh", "calf")]
TERMS = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")
CPP_TERMS = ("base_ang_vel", "projected_gravity", "keyboard_velocity_commands", "joint_pos_rel", "joint_vel_rel", "last_action")
FUNCTIONS = ("base_ang_vel", "projected_gravity", "generated_commands", "joint_pos_rel", "joint_vel_rel", "last_action")
WIDTHS = (3, 3, 3, 12, 12, 12)


class ConfigLoader(yaml.SafeLoader):
    """Read Python-tagged saved configs as data, without constructing objects."""


def tagged(loader, tag, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", tagged)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_actor(run, model):
    agent = yaml.safe_load((run/"params/agent.yaml").read_text())
    assert not agent["empirical_normalization"]
    cfg = dict(agent["policy"])
    cls_name = cfg.pop("class_name")
    if model == "Asym_PPO":
        from rsl_rl.modules import ActorCritic
        assert cls_name == "ActorCritic"
        network = ActorCritic(45, 238, 12, **cfg)
    else:
        name = "_mujoco_preview_agents"
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT/"source/dreamwaq_go2/dreamwaq_go2/tasks/manager_based/dreamwaq_go2/agents")]
        sys.modules[name] = package
        if model == "DreamWaQ_CENet":
            assert cls_name == "DreamWaQActorCritic"
            cls = importlib.import_module(name+".dreamwaq_actor_critic").DreamWaQActorCritic
            network = cls(270,238,12,**cfg)
        else:
            assert cls_name == "DreamWaQPayloadActorCritic"
            cls = importlib.import_module(name+".payload_actor_critic").DreamWaQPayloadActorCritic
            network = cls(1170,239,12,**cfg)
    saved=torch.load(run/"model_4999.pt",map_location="cpu",weights_only=True)
    torch.nn.Module.load_state_dict(network,saved["model_state_dict"],strict=True)
    return network.eval().actor, agent


def deployment(env, template):
    cfg=copy.deepcopy(template)
    cfg["joint_ids_map"]=[SDK_JOINTS.index(j) for j in TRAIN_JOINTS]
    defaults=[]
    for joint in TRAIN_JOINTS:
        matches=[v for expr,v in env["scene"]["robot"]["init_state"]["joint_pos"].items() if re.fullmatch(expr,joint)]
        assert len(matches)==1, joint
        defaults.append(matches[0])
    cfg["default_joint_pos"]=defaults
    actuator=env["scene"]["robot"]["actuators"]["base_legs"]
    cfg["stiffness"]=[actuator["stiffness"]]*12
    cfg["damping"]=[actuator["damping"]]*12
    cfg["step_dt"]=env["sim"]["dt"]*env["decimation"]
    assert cfg["step_dt"]==.02 and env["scene"]["robot"]["init_state"]["joint_vel"]=={".*":0.}
    action=env["actions"]["joint_pos"]
    assert action["scale"]==.25 and action["use_default_offset"] and action["clip"] is None
    cfg["actions"]={"JointPositionAction":dict(joint_names=[".*"],joint_ids=None,scale=[.25]*12,offset=defaults,clip=None)}
    # Preserve training's raw action mapping; do not add the old template's clamps.
    cfg["safety"]={}
    cfg["commands"]={"base_velocity":{"ranges":dict(lin_vel_x=[-.6,.6],lin_vel_y=[-.2,.2],ang_vel_z=[-.3,.3])}}
    cfg["observations"]={}
    terms=env["observations"]["policy"]
    expected=list(TERMS)
    if "history_base_ang_vel" in terms:
        expected += ["history_"+n for n in TERMS]
    actual=[n for n in terms if n in expected]
    assert actual==expected, actual
    for source in actual:
        history=source.startswith("history_")
        index=TERMS.index(source.removeprefix("history_"))
        term=terms[source]
        assert term["func"].split(":")[-1]==FUNCTIONS[index]
        assert term["scale"] is None and term["clip"] is None and not term["modifiers"]
        name=("history_" if history else "")+CPP_TERMS[index]
        cfg["observations"][name]=dict(params=term["params"],scale=None,clip=None,history_length=max(1,term["history_length"]))
    return cfg


CPP_CHECK = r'''
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/algorithms/algorithms.h"
int main(int argc,char** argv) {
  std::vector<float> frame(45);
  const std::vector<std::string> names={"base_ang_vel","projected_gravity","keyboard_velocity_commands","joint_pos_rel","joint_vel_rel","last_action"};
  const std::vector<int> widths={3,3,3,12,12,12};
  int offset=0;
  for(int j=0;j<6;++j) {
    const int begin=offset, width=widths[j];
    auto fn=[&,begin,width](isaaclab::ManagerBasedRLEnv*,YAML::Node) {
      return std::vector<float>(frame.begin()+begin,frame.begin()+begin+width);
    };
    isaaclab::observations_map()[names[j]]=fn;
    isaaclab::observations_map()["history_"+names[j]]=fn;
    offset+=width;
  }
  std::ifstream input(argv[2]);
  for(float& v:frame) input>>v;
  isaaclab::ObservationManager manager(YAML::LoadFile(argv[1])["observations"],nullptr);
  manager.reset();
  std::ofstream output(argv[3]);
  isaaclab::OrtRunner runner(argv[4]);
  std::ofstream actions_file(argv[5]);
  output<<std::setprecision(9);
  actions_file<<std::setprecision(9);
  for(int step=0;step<40;++step) {
    if(step) for(float& v:frame) input>>v;
    auto obs=manager.compute().at("obs");
    for(float v:obs) output<<v<<' ';
    output<<'\n';
    for(float v:runner.act({{"obs",obs}})) actions_file<<v<<' ';
    actions_file<<'\n';
  }
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--controller-root",type=Path,default=Path("/home/hyc/unitree_rl_lab"))
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if out.exists():
        parser.error("Use a new directory; existing exports are never overwritten")
    out.mkdir(parents=True)
    controller=args.controller_root.resolve()
    binary=controller/"deploy/robots/go2/build/go2_ctrl"
    config_path=controller/"deploy/robots/go2/config/config.yaml"
    original=yaml.safe_load(config_path.read_text())
    old_policy=Path(original["FSM"]["Velocity"]["policy_dir"])
    template=yaml.safe_load((old_policy/"params/deploy.yaml").read_text())
    protected={str(p):sha(p) for p in (binary,config_path)}
    for model,date in RUNS.items():
        p=ROOT/"logs/rsl_rl/payload_comparison"/model/date/"model_4999.pt"
        protected[str(p)]=sha(p)
    (out/"check_observations.cpp").write_text(CPP_CHECK)
    ort_root=controller/"deploy/thirdparty/onnxruntime-linux-x64-1.22.0"
    subprocess.run(["g++","-std=c++17","-O2","-I"+str(controller/"deploy/include"),
        "-I"+str(ort_root/"include"),str(out/"check_observations.cpp"),"-lyaml-cpp",
        str(ort_root/"lib/libonnxruntime.so.1"),"-Wl,-rpath,"+str(ort_root/"lib"),
        "-o",str(out/"check_observations")],check=True)
    rng=np.random.default_rng(452)
    frames=rng.normal(0,.1,(40,45)).astype(np.float32)
    frames[:,3:6]=[0,0,-1]
    frames[:,6:9]=[.6,0,.15]
    np.savetxt(out/"synthetic_frames.txt",frames,fmt="%.9g")
    results={}
    for model,date in RUNS.items():
        run=ROOT/"logs/rsl_rl/payload_comparison"/model/date
        bundle=out/model
        (bundle/"exported").mkdir(parents=True)
        (bundle/"params").mkdir()
        actor,agent=load_actor(run,model)
        env=yaml.load((run/"params/env.yaml").read_text(),Loader=ConfigLoader)
        deploy=deployment(env,template)
        deploy_path=bundle/"params/deploy.yaml"
        deploy_path.write_text(yaml.safe_dump(deploy,sort_keys=False))
        dim=sum(WIDTHS[CPP_TERMS.index(k.removeprefix("history_"))]*v["history_length"] for k,v in deploy["observations"].items())
        expected_dim={"Asym_PPO":45,"DreamWaQ_CENet":270,"PayloadEstimator":1170}[model]
        assert dim==expected_dim
        policy_path=bundle/"exported/policy.onnx"
        with torch.inference_mode():
            torch.onnx.export(actor,torch.zeros(1,dim),str(policy_path),input_names=["obs"],output_names=["actions"],opset_version=11,dynamo=False)
        onnx.checker.check_model(onnx.load(str(policy_path)))
        subprocess.run([str(out/"check_observations"),str(deploy_path),str(out/"synthetic_frames.txt"),
            str(bundle/"cpp_observations.txt"),str(policy_path),str(bundle/"cpp_actions.txt")],check=True)
        cpp=np.loadtxt(bundle/"cpp_observations.txt",dtype=np.float32)
        assert cpp.shape==(40,dim)
        history_length=0 if model=="Asym_PPO" else 5 if model=="DreamWaQ_CENet" else 25
        expected=[]
        for step,frame in enumerate(frames):
            if not history_length:
                expected.append(frame)
            else:
                hist=np.stack([frames[max(0,j)] for j in range(step-history_length+1,step+1)])
                indices=np.cumsum((0,)+WIDTHS)
                expected.append(np.concatenate([frame]+[hist[:,indices[j]:indices[j+1]].reshape(-1) for j in range(6)]))
        np.testing.assert_array_equal(cpp,np.asarray(expected))
        with torch.inference_mode():
            native=actor(torch.from_numpy(cpp)).numpy()
        exported=np.loadtxt(bundle/"cpp_actions.txt",dtype=np.float32)
        np.testing.assert_allclose(exported,native,rtol=1e-4,atol=1e-5)
        fsm=copy.deepcopy(original["FSM"])
        enabled=("Passive","FixStand","Velocity")
        fsm={k:v for k,v in fsm.items() if k=="_" or k in enabled}
        fsm["_"]={k:v for k,v in fsm["_"].items() if k in enabled}
        for name in enabled:
            for field in ("transitions","keyboard_transitions"):
                fsm[name][field]={k:v for k,v in fsm[name].get(field,{}).items() if k in enabled}
        fsm["Velocity"]["policy_dir"]=str(bundle)
        fsm["Velocity"]["lowstate_log"]["path"]=str(bundle/"lowstate_{time}.csv")
        (bundle/"config.yaml").write_text(yaml.safe_dump({"FSM":fsm},sort_keys=False))
        # /proc/self/exe must refer to the isolated copy; a symlink would use the old config.
        shutil.copy2(binary,bundle/"go2_ctrl")
        subprocess.run([str(bundle/"go2_ctrl"),"--help"],check=True,stdout=subprocess.DEVNULL)
        assert sha(bundle/"go2_ctrl")==sha(binary)
        results[model]=dict(checkpoint=str(run/"model_4999.pt"),checkpoint_sha256=sha(run/"model_4999.pt"),
            input_dim=dim,action_dim=12,training_seed=agent["seed"],
            cpp_history_and_observation_parity="40 synthetic frames exact, including reset fill",
            max_onnx_action_error=float(np.abs(exported-native).max()),
            start_command=f"cd {bundle}\n./go2_ctrl --network lo")
    assert all(sha(Path(p))==v for p,v in protected.items())
    (out/"manifest.json").write_text(json.dumps(dict(models=results,protected_sha256=protected,
        joint_order=TRAIN_JOINTS,joint_ids_map=deploy["joint_ids_map"],
        note="CPU checks only; no DDS or MuJoCo rollout performed. Manual visual sim-to-sim preview, not a formal matched-physics benchmark."),indent=2))
    print("PREPARED",out)
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
