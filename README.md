# miles-hand-on

# How to use the nodes and run the toturial

## PPT: https://docs.google.com/presentation/d/1erckPX36TyT0PRaTVElvz-VOTqCp_ZnxvjCnzEguYbk/edit?usp=sharing

## Nodes
You can use the following nodes with the password:

2 GPUs node:
- IP: `ssh -p 65308 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `zCql0YfbbqJ6t6bdiqT1`
- IP: `ssh -p 62418 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `zzJKD3p6Sxw9lpwWZAs1`
- IP: `ssh -p 50759 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `pRCxRtwi0RsV7H7aBD0G`
- IP: `ssh -p 37323 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `kwjVnxfyAGvL8iBz6jwL`
- IP: `ssh -p 33600 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `FcjhWIS8oRuoL4AXnLHp`
- IP: `ssh -p 54336 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `kZgxefjQWoS8IAINglb8`
- IP: `ssh -p 47706 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `w507AK3FzDM7UoPNPuzU`
- IP: `ssh -p 61324 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `3c2b2QMXEG7eYI20xwfr`
- IP: `ssh -p 56347 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `gRg9EFUfzIqWhEftu0Cy`
- IP: `ssh -p 48196 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `t1zI6ymAvWVsboUM9Afj`
- IP: `ssh -p 62131 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `Eu9XLUKXQLb6tH8bOYBX`
- IP: `ssh -p 37968 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `YFyuqqYMI8Z3tnvchosN`
- IP: `ssh -p 40222 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `iG0qi0GE6LuwwrNopckP`

1 GPU node:
- IP: `ssh -p 38287 root@proxy.us-ca-7.gpu-instance.novita.ai` PWD: `KUScj1uA3aqUQxmah8GS`

Note: There might be mutiple use the same nodes. Thus, you can use `nvidia-smi` to check whether any GPUs are already occupied before starting. Additionally, you can `mkdir` create a directory with your name to avoid conflicts if others use the same directory.

## Repo:
```bash
mkdir DEV_YOUR_NAME
cd DEV_YOUR_NAME
git clone https://github.com/yushengsu-thu/sglang-miles-hand-on.git
```

## Dev IDE
You can log into the above nodes to run the exmaples. To log into the node via IDE (such as Cursor or VScode), you can go to your local `.ssh/config` and add the node ip to your config, i,e.:
```bash
Host novita-gpu
    HostName proxy.us-ca-7.gpu-instance.novita.ai
    Port 40222
    User root
```
and then click <img width="1273" height="108" alt="Screenshot 2026-05-06 at 1 04 38 PM" src="https://github.com/user-attachments/assets/6b6dd85c-0350-41bb-9a74-69358b7fac8a" />

Beside, we provide `*.ipynb` (i.e. `training_lab_sglang_rl.ipynb`). You can open it and choose python environnement kerkel `/usr/bin/python`. Then you can run it.


And you will see below: 

<img width="1510" height="952" alt="Screenshot 2026-05-06 at 11 40 28 AM" src="https://github.com/user-attachments/assets/eb8518cc-529f-4cd9-bad2-b68604c965e8" />
