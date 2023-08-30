Benchmark to measure the training throughput of a 7B LoRA model and a 1.4B baseline on standard EC2 instances. Uses Terraform to manage instances.

# Requirements

To run this benchmark, you'll need to have an AWS account. You'll also need sufficient quota to spin up the number of `g5.xlarge` instances that you want to test with.

You'll also need to obtain access to the [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model on Hugging Face. If you haven't already, you'll need to create an account on Hugging Face and follow the instructions on the model's page. Then, you'll need an HF access token for your account, which you can generate from [this page](https://huggingface.co/settings/tokens).

# How to run

## Install dependencies

- Install [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
- Run `pip install -r requirements.txt`
- Install the AWS CLI and [configure it with your credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)

## Spin up EC2 instances

```bash
terraform apply -var "public_key=$(cat $path_to_your_id_rsa_dot_pub_or_equivalent)"
```

## Run the benchmark

```bash
$ python run.py --ssh-privkey-path $path_to_your_id_rsa_or_equivalent --hf-auth-token $your_hf_auth_token
```

After a while, after training runs are kicked off, you should see output like:

```
Node 0: Average step time ____ ms...
Node 0: Average step time ____ ms...
Node 0: Average step time ____ ms...
Node 0: Average step time ____ ms...
...
```

## Tear down the instances

Make sure to tear down your Terraform stack once done, to avoid paying more than necessary for GPU instances.

```bash
$ terraform destroy -var "public_key="
```
