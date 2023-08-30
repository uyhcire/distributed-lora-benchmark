provider "aws" {
  region = "us-west-2"
}

variable "public_key" {
  description = "The public SSH key to access instances with."
  type        = string
}

# Create a VPC
resource "aws_vpc" "my_vpc" {
  cidr_block         = "10.0.0.0/16"
  enable_dns_support = true
}

# Create a subnet
resource "aws_subnet" "my_subnet" {
  cidr_block = "10.0.1.0/24"
  vpc_id     = aws_vpc.my_vpc.id
  # Some availability zones in us-west-2 do not have g5.xlarge instances available.
  availability_zone = "us-west-2a"
  # Ensure that we can access EC2 instances over SSH.
  map_public_ip_on_launch = true
}

# Create a security group
resource "aws_security_group" "allow_ssh_and_icmp" {
  name        = "allow_ssh_and_icmp"
  description = "Allow SSH, ICMP, and Intra-VPC traffic"
  vpc_id      = aws_vpc.my_vpc.id

  # SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # ICMP
  ingress {
    from_port   = -1
    to_port     = -1
    protocol    = "icmp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow all traffic within VPC - required for distributed training to work.
  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "udp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

module "key_pair" {
  source = "terraform-aws-modules/key-pair/aws"

  key_name   = "my_key_pair"
  public_key = var.public_key
}

resource "aws_instance" "my_instance" {
  count         = 1
  instance_type = "g5.xlarge"

  # Deep Learning AMI GPU PyTorch 2.0.1 (Amazon Linux 2)
  ami = "ami-0574dd0eaf225e833"

  key_name = "my_key_pair"

  subnet_id = aws_subnet.my_subnet.id

  vpc_security_group_ids = [aws_security_group.allow_ssh_and_icmp.id]

  tags = {
    Name = "Node-${count.index}"
  }

  depends_on = [
    aws_subnet.my_subnet,
    aws_security_group.allow_ssh_and_icmp
  ]
}

resource "aws_internet_gateway" "my_gateway" {
  vpc_id = aws_vpc.my_vpc.id
}

resource "aws_route_table" "my_route_table" {
  vpc_id = aws_vpc.my_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.my_gateway.id
  }
}

resource "aws_route_table_association" "my_route_table_association" {
  subnet_id      = aws_subnet.my_subnet.id
  route_table_id = aws_route_table.my_route_table.id
}

# Emit the IPs of the instances

output "instance_public_ips" {
  value = aws_instance.my_instance.*.public_ip
}

output "instance_private_ips" {
  value = aws_instance.my_instance.*.private_ip
}