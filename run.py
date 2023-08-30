import argparse
import os
import tempfile
import json
import subprocess

import paramiko
from tqdm import tqdm


MAX_NODES = 1
print(f"Using node count: {MAX_NODES}")


parser = argparse.ArgumentParser(
    description="Run the training benchmark on a set of Terraform-managed instances."
)
parser.add_argument(
    "--ssh-privkey-path", type=str, required=True, help="Path to your SSH private key."
)
parser.add_argument(
    "--hf-auth-token", type=str, required=True, help="Your Hugging Face auth token."
)
args = parser.parse_args()


def upload(client, local_path, remote_path):
    """Uploads a file to a remote machine."""
    sftp = client.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()


# Run the Terraform output command and capture the output
with tempfile.NamedTemporaryFile(mode="r+", delete=False) as tf_output:
    subprocess.run(["terraform", "output", "-json"], stdout=tf_output)
    tf_output.seek(0)
    terraform_data = json.load(tf_output)

# Extract IPs from the JSON and populate the nodes list
instance_private_ips = terraform_data["instance_private_ips"]["value"]
instance_public_ips = terraform_data["instance_public_ips"]["value"]

nodes = [
    {"public_ip": pub_ip, "private_ip": priv_ip}
    for pub_ip, priv_ip in zip(instance_public_ips, instance_private_ips)
]
assert len(nodes) >= MAX_NODES
nodes = nodes[:MAX_NODES]

# Use the first listed node as the master node
master_ip = nodes[0]["private_ip"]
master_port = "29500"  # any free port should work
world_size = len(nodes)

ssh_clients = []

# Create SSH clients for each node and connect
for node in tqdm(nodes, desc="Uploading data"):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        node["public_ip"],
        username="ec2-user",
        key_filename=args.ssh_privkey_path,
    )
    ssh_clients.append(client)

    # Upload training script and data to each node
    upload(client, "train.py", "/home/ec2-user/train.py")
    upload(client, "all_examples.json", "/home/ec2-user/all_examples.json")

# Execute commands on each node to initialize distributed training
for i, client in tqdm(list(enumerate(ssh_clients)), desc="Starting training scripts"):
    env_vars = f"NCCL_DEBUG=INFO MASTER_ADDR={master_ip} MASTER_PORT={master_port} WORLD_SIZE={world_size} RANK={i}"
    run_command = (
        "source activate pytorch; "
        + "pip3 install transformers accelerate peft; "
        + f"{env_vars} python3 -u /home/ec2-user/train.py --hf-auth-token {args.hf_auth_token}"
    )

    # Run the command in the background so that SSH session doesn't wait for it to complete
    stdin, stdout, stderr = client.exec_command(
        "killall -q python3; "
        + "rm -rf /tmp/training.log; "
        + f'nohup sh -c "{run_command}" > /tmp/training.log 2>&1 &'
    )

print("Distributed training initialized on all nodes.")

stdin, stdout, stderr = ssh_clients[0].exec_command(f"tail -f /tmp/training.log")
while True:
    line = stdout.readline().strip()
    print(f"Node 0: {line}")
