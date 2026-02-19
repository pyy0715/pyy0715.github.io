---
date: 2026-02-15 00:00:00
layout: post
title: "Bootstrapping Air-gapped Kubernetes clusters — 0"
type: Guide
math: true

category: DevOps
tags:
  - Kubernetes
  - DevOps
  - Kubespray
author: pyy0715
---

> Cludneta@K8S-Deploy Week6

## Overview

폐쇄망(Air-Gap) 환경에서 Kubernetes 클러스터를 배포하는 작업은 온라인 환경과 완전히 다른 접근 방식이 필요합니다. 온라인 환경에서는 kubeadm init한 줄이면 Control Plane 컨테이너 이미지가 자동으로 pull되지만, 폐쇄망에서는 그 이미지를 어디서, 어떤 경로로, 어떤 인증 방식으로 가져올 것인지를 사전에 설계해야 합니다. 이 글에서는 Kubespray 오프라인 클러스터 배포의 전 단계에 해당하는 인프라 환경 구성을 다룹니다.

## Security Boundary Design Principles

지난 실습 환경은 admin 서버 1대가 인터넷에 직접 연결된 상태에서 Kubespray를 실행하고, k8s-node가 admin을 통해 간접적으로 외부 리소스에 접근하는 구조였습니다.

이 구성은 학습 효율 측면에서는 합리적이지만, 실제 기업 폐쇄망 아키텍처와는 중요한 차이가 존재합니다.

![Network Topology](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*KTZyUHmyRDKVn_gjGUJjbQ.png) *Network Topology*

실무에서는 외부 인터넷에 접근할 수 있는 서버(Bastion)와 내부 인프라를 관리하는 서버(Admin)가 물리적/논리적으로 분리되어 있으며, 두 영역 사이에는 방화벽 정책 승인 절차가 개입합니다. 이 분리를 무시하고 단일 서버에서 모든 역할을 수행하면, 실무 전환 시 파일 전송 경로 설계, 서비스 기동 순서, 네트워크 격리 검증 등에서 문제가 발생할 수 있습니다.

이러한 계층적 분리의 핵심은 보안 경계의 명시적 설정입니다. 일반적인 기업망에서는 외부 인터넷과 DMZ 사이에 외부 방화벽이, DMZ와 내부망 사이에 내부 방화벽이 위치합니다.

Bastion Server는 DMZ에 속하며, 외부 방화벽 정책 승인을 통해 특정 포트와 도메인에 한정된 외부 접근이 허용됩니다. Admin Server는 내부망에 속하며, Bastion으로부터 파일을 수신하는 것 외에는 외부와의 직접 통신이 차단됩니다.

이 구조가 보장하는 것은 최소 권한 원칙(Principle of Least Privilege)입니다. 외부 인터넷에 접근할 수 있는 노드를 Bastion 1대로 한정함으로써, 공격 표면(Attack Surface)을 최소화하고 내부망 경로를 제한합니다.

Vagrant 실습에서 이 구조를 구현하기 위해서, Bastion VM만 VirtualBox NAT 인터페이스를 통해 외부 통신이 가능하고, Admin과 K8s Node VM은 NAT 인터페이스의 default route를 제거하여 외부 통신을 완전히 차단해야 합니다. 파일 전달은 Bastion → Admin 방향으로 rsync또는 scp를 사용하며, Admin → K8s Node 방향으로는 HTTP 기반 내부 서비스(Registry, YUM Repo 등)를 통해 이루어집니다.

### VirtualBox Network Modes

Vagrant와 VirtualBox를 사용하여 폐쇄망을 설계하기 위해서는, 각 VM에 할당되는 네트워크 인터페이스의 종류와 역할을 명확히 설계해야 합니다. VirtualBox는 VM당 최대 4개의 네트워크 어댑터를 제공하며, 각 어댑터의 모드에 따라 트래픽 흐름이 달라집니다.

1. **NAT**: VM의 외부 인터넷 접근
2. **Host-only**: Host-VM 간 통신, VM 간 통신
3. **Internal Network**: VM 간 격리 통신
4. **Bridged**: 물리 네트워크 직접 연결

Vagrant는 기본적으로 첫 번째 어댑터(Adapter 1)를 NAT 모드로 설정합니다. 이 인터페이스(enp0s8)는 10.0.2.0/24 대역을 사용하며, VirtualBox 내장 DHCP를 통해 IP를 할당받고 default route를 자동 생성합니다. vagrant ssh 접속도 이 NAT 인터페이스의 포트 포워딩(guest 22 → host 60000번대)을 통해 이루어집니다.

두 번째 어댑터(Adapter 2)는 Vagrantfile에서 private_network로 지정하면 Host-only 모드로 동작합니다. 이 인터페이스(enp0s9)는 192.168.10.0/24 대역에 고정 IP를 부여받으며, 같은 Host-only 네트워크에 속한 VM 간 통신이 가능합니다.

![VirtualBox Network Modes](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*tCJkagajduuZ2NmtxR2tcw.png) *VirtualBox Network Modes*

따라서 Bastion VM은 NAT 인터페이스의 default route를 유지하여 외부 인터넷에 접근하고, Admin과 K8s Node VM은 NAT 인터페이스의 default route를 제거하여 외부 통신을 차단합니다. 모든 VM은 Host-only 네트워크(192.168.10.0/24)를 통해 상호 통신하며, 파일 전달은 이 경로를 통해 Bastion->Admin 방향으로 rsync 를 사용합니다.

기존 실습의 init_cfg.sh에서 수행하는 NAT 인터페이스(enp0s8) Default route 제거는 nmcli connection modify enp0s9 ipv4.never-default yes설정을 통해 Host-only 인터페이스가 default gateway를 생성하지 않도록 하는 것이었습니다.

이를 확장하여, Admin과 K8s Node에서는 enp0s8 연결 자체의 비활성화는 두 단계로 이뤄집니다.

```bash
# connection down은 실행하지 않음
nmcli connection modify enp0s8 connection.autoconnect no
```

프로비저닝 중에는 autoconnect no만 설정하고, 프로비저닝 완료 후 reboot로 VM을 재부팅하면 비활성화가 적용됩니다.

이는 Vagrant의 SSH 메커니즘 때문입니다. Vagrant는 VM 내부에서 프로비저닝 스크립트를 실행할 때, NAT 인터페이스(enp0s8)의 포트 포워딩(guest 22 → host 60000번대)을 통해 SSH 세션을 유지합니다.

만약 스크립트 내에서 nmcli connection down enp0s8을 실행하면, **바로 그 시점에 Vagrant가 사용 중인 SSH 연결이 끊어지게 됩니다.** 결과적으로 프로비저닝이 비정상 종료되고, 이후 TASK들이 실행되지 않게 됩니다.

따라서 Bastion에서 Host-only 네트워크를 통해 직접 reboot 명령을 전송하는 방식을 사용합니다.

autoconnect no만 설정하면 현재 활성화된 연결에는 영향을 주지 않으므로 프로비저닝이 정상 완료되고, VM 재부팅 시점에 NetworkManager가 enp0s8.nmconnection의 autoconnect=false를 읽어 해당 인터페이스를 활성화하지 않도록 합니다.

## Lab Environment with Vagrant

```ruby
# Base Image  https://portal.cloud.hashicorp.com/vagrant/discover/bento/rockylinux-10.0
BOX_IMAGE = "bento/rockylinux-10.0"
BOX_VERSION = "202510.26.0"
N = 2

Vagrant.configure("2") do |config|

  # Nodes
  (1..N).each do |i|
    config.vm.define "k8s-node#{i}" do |subconfig|
      subconfig.vm.box = BOX_IMAGE
      subconfig.vm.box_version = BOX_VERSION
      subconfig.vm.provider "virtualbox" do |vb|
        vb.customize ["modifyvm", :id, "--groups", "/Kubespray-offline-Lab"]
        vb.customize ["modifyvm", :id, "--nicpromisc2", "allow-all"]
        vb.name = "k8s-node#{i}"
        vb.cpus = 4
        vb.memory = 2048
        vb.linked_clone = true
      end
      subconfig.vm.host_name = "k8s-node#{i}"
      subconfig.vm.network "private_network", ip: "192.168.10.1#{i}"
      subconfig.vm.network "forwarded_port", guest: 22, host: "6000#{i}", auto_correct: true, id: "ssh"
      subconfig.vm.synced_folder "./", "/vagrant", disabled: true
      subconfig.vm.provision "shell", path: "init_cfg.sh", args: [ N ]
    end
  end

  # Admin Node
  config.vm.define "admin" do |subconfig|
    subconfig.vm.box = BOX_IMAGE
    subconfig.vm.box_version = BOX_VERSION
    subconfig.vm.provider "virtualbox" do |vb|
      vb.customize ["modifyvm", :id, "--groups", "/Kubespray-offline-Lab"]
      vb.customize ["modifyvm", :id, "--nicpromisc2", "allow-all"]
      vb.name = "admin"
      vb.cpus = 4
      vb.memory = 2048
      vb.linked_clone = true
    end
    subconfig.vm.host_name = "admin"
    subconfig.vm.network "private_network", ip: "192.168.10.10"
    subconfig.vm.network "forwarded_port", guest: 22, host: "60010", auto_correct: true, id: "ssh"
    subconfig.vm.synced_folder "./", "/vagrant", disabled: true
    subconfig.vm.provision "shell", path: "admin.sh", args: [ N ]
  end

  # Bastion
  config.vm.define "bastion" do |subconfig|
    subconfig.vm.box = BOX_IMAGE
    subconfig.vm.box_version = BOX_VERSION
    subconfig.vm.provider "virtualbox" do |vb|
      vb.customize ["modifyvm", :id, "--groups", "/Kubespray-offline-Lab"]
      vb.customize ["modifyvm", :id, "--nicpromisc2", "allow-all"]
      vb.name = "bastion"
      vb.cpus = 4
      vb.memory = 2048
      vb.linked_clone = true
    end
    subconfig.vm.host_name = "bastion"
    subconfig.vm.network "private_network", ip: "192.168.10.20"
    subconfig.vm.network "forwarded_port", guest: 22, host: "60020", auto_correct: true, id: "ssh"
    subconfig.vm.synced_folder "./", "/vagrant", disabled: true
    subconfig.vm.disk :disk, size: "120GB", primary: true
    subconfig.vm.provision "shell", path: "bastion.sh", args: [ N ]
  end
end
```

Bastion VM은 두 가지 네트워크 인터페이스를 보유합니다. enp0s8(NAT)은 VirtualBox가 자동 할당하는 10.0.2.0/24대역으로, default route가 유지되어 외부 인터넷 접근이 가능합니다. enp0s9(Host-only)는 192.168.10.20이 고정 할당되어 내부망과의 통신 경로를 제공합니다. 120GB 디스크는 reposync로 동기화할 RPM 패키지(BaseOS 4.8GB + AppStream 13GB + Extras 67MB), 컨테이너 이미지, Python 패키지 등을 저장하기 위해 필요합니다.

Bastion의 프로비저닝 스크립트( bastion.sh)는 외부 리소스 다운로드와 Admin 서버로의 파일 전송을 담당합니다.

```bash
#!/usr/bin/env bash

echo ">>>> Bastion Node Initial Config Start <<<<"

echo "[TASK 1] Change Timezone and Enable NTP"
timedatectl set-local-rtc 0
timedatectl set-timezone Asia/Seoul

echo "[TASK 2] Disable firewalld and selinux"
systemctl disable --now firewalld >/dev/null 2>&1
setenforce 0
sed -i 's/^SELINUX=enforcing/SELINUX=permissive/' /etc/selinux/config

echo "[TASK 3] Setting Local DNS Using Hosts file"
sed -i '/^127\.0\.\(1\|2\)\.1/d' /etc/hosts
echo "192.168.10.20 bastion" >> /etc/hosts
echo "192.168.10.10 admin"   >> /etc/hosts
for (( i=1; i<=$1; i++ )); do echo "192.168.10.1$i k8s-node$i" >> /etc/hosts; done

echo "[TASK 4] Delete default routing - enp0s9 NIC"
nmcli connection modify enp0s9 ipv4.never-default yes
nmcli connection up enp0s9 >/dev/null 2>&1

echo "[TASK 5] Install packages"
dnf install -y python3-pip git sshpass cloud-utils-growpart createrepo dnf-plugins-core podman >/dev/null 2>&1

echo "[TASK 6] Install Helm"
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | DESIRED_VERSION=v3.20.0 bash >/dev/null 2>&1

echo "[TASK 7] Increase Disk Size (120GB)"
growpart /dev/sda 3 >/dev/null 2>&1
xfs_growfs /dev/sda3 >/dev/null 2>&1

echo "[TASK 8] Setting SSHD"
echo "root:qwe123" | chpasswd

cat << EOF >> /etc/ssh/sshd_config
PermitRootLogin yes
PasswordAuthentication yes
EOF
systemctl restart sshd >/dev/null 2>&1

echo "[TASK 9] Setting SSH Key (Bastion → Admin, K8s Nodes)"
ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa >/dev/null 2>&1

# Bastion
sshpass -p 'qwe123' ssh-copy-id -o StrictHostKeyChecking=no root@192.168.10.20 >/dev/null 2>&1

# Admin
sshpass -p 'qwe123' ssh-copy-id -o StrictHostKeyChecking=no root@192.168.10.10 >/dev/null 2>&1
ssh -o StrictHostKeyChecking=no root@admin hostname >/dev/null 2>&1

# K8s Nodes
for (( i=1; i<=$1; i++ )); do
  sshpass -p 'qwe123' ssh-copy-id -o StrictHostKeyChecking=no root@192.168.10.1$i >/dev/null 2>&1
  ssh -o StrictHostKeyChecking=no root@k8s-node$i hostname >/dev/null 2>&1
done

echo "[TASK 10] Install K9s"
CLI_ARCH=amd64
if [ "$(uname -m)" = "aarch64" ]; then CLI_ARCH=arm64; fi
wget -P /tmp https://github.com/derailed/k9s/releases/latest/download/k9s_linux_${CLI_ARCH}.tar.gz >/dev/null 2>&1
tar -xzf /tmp/k9s_linux_${CLI_ARCH}.tar.gz -C /tmp
chown root:root /tmp/k9s
mv /tmp/k9s /usr/local/bin/
chmod +x /usr/local/bin/k9s

echo "[TASK 11] ETC"
echo "sudo su -" >> /home/vagrant/.bashrc

echo ">>>> Bastion Node Initial Config End <<<<"
```

이후 admin.sh 에서 프로비저닝 시점에는 enp0s8(NAT)이 활성 상태이므로 dnf install 등 인터넷이 필요한 작업이 가능하며, 모든 설정 완료 후 enp0s8을 비활성화하여 인터넷을 차단합니다.

이후 Admin에서 필요한 모든 소프트웨어(bind, nginx, devpi-server 등)는 Bastion이 전송한 /data/ 디렉터리의 RPM 파일과 Python wheel로 오프라인 설치합니다.

```bash
#!/usr/bin/env bash

echo ">>>> Admin Node Initial Config Start <<<<"

echo "[TASK 1] Change Timezone and Enable NTP"
timedatectl set-local-rtc 0
timedatectl set-timezone Asia/Seoul

echo "[TASK 2] Disable firewalld and selinux"
systemctl disable --now firewalld >/dev/null 2>&1
setenforce 0
sed -i 's/^SELINUX=enforcing/SELINUX=permissive/' /etc/selinux/config

echo "[TASK 3] Setting Local DNS Using Hosts file"
sed -i '/^127\.0\.\(1\|2\)\.1/d' /etc/hosts
echo "192.168.10.20 bastion" >> /etc/hosts
echo "192.168.10.10 admin"   >> /etc/hosts
for (( i=1; i<=$1; i++ )); do echo "192.168.10.1$i k8s-node$i" >> /etc/hosts; done

echo "[TASK 4] Install packages"
dnf install -y python3-pip git sshpass >/dev/null 2>&1

echo "[TASK 5] Setting SSHD"
echo "root:qwe123" | chpasswd

cat << EOF >> /etc/ssh/sshd_config
PermitRootLogin yes
PasswordAuthentication yes
EOF
systemctl restart sshd >/dev/null 2>&1

echo "[TASK 6] Setting SSH Key (Admin → K8s Nodes)"
ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa >/dev/null 2>&1

# Admin
sshpass -p 'qwe123' ssh-copy-id -o StrictHostKeyChecking=no root@192.168.10.10 >/dev/null 2>&1

# K8s Nodes
for (( i=1; i<=$1; i++ )); do
  sshpass -p 'qwe123' ssh-copy-id -o StrictHostKeyChecking=no root@192.168.10.1$i >/dev/null 2>&1
  ssh -o StrictHostKeyChecking=no root@k8s-node$i hostname >/dev/null 2>&1
done

echo "[TASK 7] Delete default routing - enp0s9 NIC"
nmcli connection modify enp0s9 ipv4.never-default yes
nmcli connection up enp0s9 >/dev/null 2>&1

echo "[TASK 8] Disable NAT - enp0s8 NIC"
nmcli connection modify enp0s8 connection.autoconnect no

echo "[TASK 9] ETC"
echo "sudo su -" >> /home/vagrant/.bashrc

echo ">>>> Admin Node Initial Config End <<<<"
```

K8s Node VM은 기존 init_cfg.sh를 그대로 사용하지만 enp0s8 (NAT) 를 비활성화하여 외부 통신을 완전히 차단합니다. K8s Node에서 필요한 모든 리소스(RPM 패키지, 컨테이너 이미지)는 Admin이 제공하는 내부 서비스(YUM Mirror, Container Registry)를 통해 공급받습니다.

```bash
#!/usr/bin/env bash

echo ">>>> Initial Config Start <<<<"


echo "[TASK 1] Change Timezone and Enable NTP"
timedatectl set-local-rtc 0
timedatectl set-timezone Asia/Seoul


echo "[TASK 2] Disable firewalld and selinux"
systemctl disable --now firewalld >/dev/null 2>&1
setenforce 0
sed -i 's/^SELINUX=enforcing/SELINUX=permissive/' /etc/selinux/config


echo "[TASK 3] Disable and turn off SWAP & Delete swap partitions"
swapoff -a
sed -i '/swap/d' /etc/fstab
sfdisk --delete /dev/sda 2 >/dev/null 2>&1
partprobe /dev/sda >/dev/null 2>&1


echo "[TASK 4] Config kernel & module"
cat << EOF > /etc/modules-load.d/k8s.conf
overlay
br_netfilter
vxlan
EOF
modprobe overlay >/dev/null 2>&1
modprobe br_netfilter >/dev/null 2>&1

cat << EOF > /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF
sysctl --system >/dev/null 2>&1


echo "[TASK 5] Setting Local DNS Using Hosts file"
sed -i '/^127\.0\.\(1\|2\)\.1/d' /etc/hosts
echo "192.168.10.20 bastion" >> /etc/hosts
echo "192.168.10.10 admin"   >> /etc/hosts
for (( i=1; i<=$1; i++  )); do echo "192.168.10.1$i k8s-node$i" >> /etc/hosts; done


echo "[TASK 6] Delete default routing - enp0s9 NIC"
nmcli connection modify enp0s9 ipv4.never-default yes
nmcli connection up enp0s9 >/dev/null 2>&1

echo "[TASK 7] Disable NAT - enp0s8 NIC"
nmcli connection modify enp0s8 connection.autoconnect no


echo "[TASK 8] Setting SSHD"
echo "root:qwe123" | chpasswd

cat << EOF >> /etc/ssh/sshd_config
PermitRootLogin yes
PasswordAuthentication yes
EOF
systemctl restart sshd >/dev/null 2>&1


echo "[TASK 9] Install packages"
dnf install -y python3-pip git >/dev/null 2>&1


echo "[TASK 10] ETC"
echo "sudo su -" >> /home/vagrant/.bashrc


echo ">>>> Initial Config End <<<<"
```

프로비저닝이 완료되었다면 NetworkManager가 인터페이스를 활성화하지 않도록 재부팅을 합니다. 재부팅 후에는 인터넷이 비활성화되므로 vagrant ssh admin은 동작하지 않게 됩니다. NAT 포트 포워딩 경로가 비활성화되었기 때문입니다.

```bash
# vagrant up 완료 후, Bastion에서 실행
ssh admin "reboot"
ssh k8s-node1 "reboot"
ssh k8s-node2 "reboot"
```

이후 접속은 Host OS에서 ssh root@192.168.10.10로 직접 접속하거나, vagrant ssh bastion 후, bastion에서 admin으로 SSH 접속하는 방식을 사용합니다.

### Network Verification

재부팅 후, 가장 먼저 확인해야 할 것은 인터넷 격리가 의도한 대로 적용되었는지입니다. enp0s8(NAT)이 비활성화되면 두 가지 변화가 나타납니다. 인터페이스에 IP가 할당되지 않고, 10.0.2.0/24 대역의 default route가 라우팅 테이블에서 사라집니다. Bastion에서 각 VM에 접속하여 이 두 가지를 확인합니다.

```bash
for host in admin k8s-node1 k8s-node2; do
  echo "===== $host ====="
  ssh "$host" "ip -br addr"
done
```

![ip -br addr output](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*-UJVLl92VwzRKhWlC9E-ZQ.png) *ip -br addr output*

enp0s8은 현재 어떤 네트워크에도 연결되지 않은 유휴 인터페이스입니다. 운영체제 관점에서는 장치가 활성화(UP)되어 있지만 IP 주소와 연결 프로파일이 없습니다. 즉, 전원은 들어와 있으나 케이블이 연결되지 않은 랜포트와 같은 상태입니다.

![nmcli device status](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*l0Bk5nuDD9x3VfGB6zRsRg.png) *ssh admin "nmcli device status"*

## Resource Transfer (Bastion → Admin)

Bastion은 인터넷에 접근할 수 있는 유일한 노드로서 외부 리소스를 수집하고, Admin은 이를 받아 내부망에 서비스로 배포하는 구조입니다.

Bastion에서 수집해야 할 리소스는 크게 네 가지 카테고리로 분류됩니다.

- **Container Images**: Private Registry 구동용
- **RPM Repositories**: YUM Mirror 구성용
- **Python Packages**: devpi-server 구동용
- **RPM Packages**: bind, nginx 등 서비스 설치용

먼저 리소스 저장 디렉터리를 생성하고, 컨테이너 이미지를 다운로드합니다.

```bash
# Bastion에서 실행
mkdir -p /data/{images,repos/rocky/10,pypi,rpms}

# Container Images
podman pull docker.io/library/registry:latest
podman pull docker.io/library/alpine:latest
podman save -o /data/images/registry.tar docker.io/library/registry:latest
podman save -o /data/images/alpine.tar docker.io/library/alpine:latest
```

다음으로 RPM 저장소를 동기화합니다. 이 작업은 BaseOS와 AppStream의 크기가 크므로 10~15분 정도 소요됩니다.

```bash
# RPM Repositories (10-15 min)
dnf reposync --repoid=baseos    --download-metadata -p /data/repos/rocky/10
dnf reposync --repoid=appstream --download-metadata -p /data/repos/rocky/10
dnf reposync --repoid=extras    --download-metadata -p /data/repos/rocky/10
```

Python 패키지와 서비스 설치용 RPM을 다운로드합니다. Kubespray는 jmespath와 netaddr을 필수 의존성으로 요구하며, devpi-server는 내부 PyPI 미러 구성을 위해 필요합니다.

```bash
# Python Packages
pip download jmespath netaddr -d /data/pypi
pip download devpi-server devpi-client devpi-web -d /data/pypi

# RPM Packages for offline install
dnf download --resolve --destdir=/data/rpms bind bind-utils nginx chrony
```

rsync로 전체 디렉터리를 전송합니다.

```bash
# Bastion → Admin 전송
rsync -avz --progress /data/ root@admin:/data/

# 데이터 확인
ssh admin "du -sh /data/*"
```

## Infrastructure Services

### NTP Server-Client

Kubernetes 컴포넌트 간 통신에 사용되는 TLS 인증서는 유효 기간이 명시되어 있으며, 노드 간 시간 차이가 이 범위를 벗어나면 인증서 검증이 실패합니다. etcd의 Raft 합의 알고리즘에서도 타임아웃 기반 판단을 수행하므로, 노드 간 시간 편차가 크면 불필요한 리더 선출이 반복되어 클러스터 안정성이 저하됩니다.

폐쇄망(air-gapped) 환경에서는 admin이 외부 NTP 서버에서 시간을 가져올 수 없지만, local stratum 10설정 덕분에 admin의 로컬 시계를 기준으로 k8s 노드들이 시간을 동기화받을 수 있습니다.

따라서 ssh admin 에서 아래와 같이 적용합니다.

```bash
# 백업
cp /etc/chrony.conf /etc/chrony.bak

# chrony 설정
cat << EOF > /etc/chrony.conf
# 내부망 클라이언트 접근 허용
allow 192.168.10.0/24

# 외부망 단절 시 로컬 시계 기준으로 서비스 제공
local stratum 10

logdir /var/log/chrony
EOF

# 재시작
systemctl restart chronyd.service
```

이후, 각 k8s-node는 iburst 옵션을 통해 admin 서버와 동기화하도록 설정합니다.

```bash
cp /etc/chrony.conf /etc/chrony.bak

cat << EOF > /etc/chrony.conf
server 192.168.10.10 iburst
logdir /var/log/chrony
EOF

systemctl restart chronyd.service
```

노드들이 사용 중인 NTP 서버를 아래와 같이 확인할 수 있습니다.

![chronyc sources](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*lNFH64-tBASATcTl-0B6HA.png) *chronyc sources -v*

Admin에서 자신의 NTP 서버를 사용하는 클라이언트를 확인할 수 있습니다.

![chronyc clients](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*DwMTWswGKPhk_2kLDwskag.png) *chronyc clients*

### DNS Server — Client

Kubespray는 내부 동작 중 다수의 도메인 기반 접근을 수행합니다. 폐쇄망에서 이 도메인들을 내부 IP로 해석하려면 DNS 서버가 필수이며, /etc/hosts 파일만으로는 모든 시나리오를 커버할 수 없습니다. 특히 컨테이너 내부에서 실행되는 프로세스는 호스트의 /etc/hosts를 참조하지 않는 경우가 많기 때문입니다.

DNS 설정을 위해 Bastion이 전송한 RPM으로 오프라인 설치합니다. 의존성 패키지는 아래와 같습니다.

![ls -lh /data/rpms](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*nTmAeQhRMyid4HAYkXC4AA.png) *ls -lh /data/rpms*

— disablerepo=* 옵션을 추가하면 원격 리포지토리(baseos 등)를 비활성화하고 오직 로컬 RPM 파일만 사용하여 설치하게 됩니다.

```bash
dnf localinstall -y /data/rpms/*.rpm --disablerepo=*
```

이를 통해 bind가 설치되었다면, Admin은 내부망 DNS 서버로 동작하고, K8s Node는 Admin을 DNS 서버로 참조하도록 설정합니다.

```bash
# 백업
cp /etc/named.conf /etc/named.bak

# BIND 설정
cat << EOF > /etc/named.conf
options {
        listen-on port 53 { any; };
        listen-on-v6 port 53 { ::1; };
        directory       "/var/named";
        dump-file       "/var/named/data/cache_dump.db";
        statistics-file "/var/named/data/named_stats.txt";
        memstatistics-file "/var/named/data/named_mem_stats.txt";
        secroots-file   "/var/named/data/named.secroots";
        recursing-file  "/var/named/data/named.recursing";
        allow-query     { 127.0.0.1; 192.168.10.0/24; };
        allow-recursion { 127.0.0.1; 192.168.10.0/24; };

        recursion yes;

        # 폐쇄망에서는 DNSSEC validation 비활성화 (root 키를 동기화할 수 없으므로)
        dnssec-validation no;

        managed-keys-directory "/var/named/dynamic";
        pid-file "/run/named/named.pid";
        session-keyfile "/run/named/session.key";
        include "/etc/crypto-policies/back-ends/bind.config";
};

logging {
        channel default_debug {
                file "data/named.run";
                severity dynamic;
        };
};

zone "." IN {
        type hint;
        file "named.ca";
};

include "/etc/named.rfc1912.zones";
EOF

# 설정 검증
named-checkconf /etc/named.conf

# 서비스 시작
systemctl enable --now named

# DNS 자기 자신 사용 설정
echo "nameserver 192.168.10.10" > /etc/resolv.conf
```

이후 각 노드에서 NetworkManager가 /etc/resolv.conf를 자동으로 덮어쓰는 것을 방지합니다.

```bash
# NetworkManager가 DNS를 자동으로 관리하지 않도록 설정
cat << EOF > /etc/NetworkManager/conf.d/99-dns-none.conf
[main]
dns=none
EOF

# NetworkManager 재시작
systemctl restart NetworkManager

# Admin을 DNS 서버로 설정
echo "nameserver 192.168.10.10" > /etc/resolv.conf

# DNS 동작 테스트
ping -c 2 admin
nslookup admin
```

### Local (Mirror) YUM/DNF Repository

Kubespray는 K8s Node에서 containerd, runc, conntrack-tools 등의 RPM 패키지를 설치합니다. 폐쇄망에서는 외부 저장소에 접근할 수 없으므로, Bastion이 동기화한 저장소를 Admin의 nginx를 통해 HTTP로 제공합니다.

```bash
cat <<EOF > /etc/nginx/conf.d/repos.conf
server {
    listen 80;
    server_name repo-server;

    location /rocky/10/ {
        autoindex on;
        autoindex_exact_size off;
        autoindex_localtime on;
        root /data/repos;
    }
}
EOF

# nginx 시작
systemctl enable --now nginx

# 접속 테스트
curl http://192.168.10.10/rocky/10/
```

nginx가 로컬 파일 시스템의 /data/repos 디렉토리를 HTTP로 공유해 주는 역할을 수행합니다.

![tree output](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*WFpMLp77k32SHYVTXD6pDw.png) *tree -L 2 /data/repos/rocky/10*

이를 통해 admin과 노드 모두 외부 인터넷 저장소 대신 내부 패키지 저장소만 사용하도록 구성합니다. 이는 실제로 아래와 같이 동작하게 됩니다.

![RHEL Repository](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*EXdn30hYNX3pDIR0YmBxwA.png) *[[Linux] 사설 RHEL Repository 만들기](https://velog.io/@ljk0509/Linux-%EC%82%AC%EC%84%A4-RHEL-Repository-%EB%A7%8C%EB%93%A4%EA%B8%B0)*

```bash
# 기존 저장소 설정 백업
mkdir -p /etc/yum.repos.d/backup
mv /etc/yum.repos.d/*.repo /etc/yum.repos.d/backup/

# 내부 Rocky Linux 저장소 설정
cat <<EOF > /etc/yum.repos.d/internal-rocky.repo
[internal-baseos]
name=Internal Rocky 10 BaseOS
baseurl=http://192.168.10.10/rocky/10/baseos
enabled=1
gpgcheck=0

[internal-appstream]
name=Internal Rocky 10 AppStream
baseurl=http://192.168.10.10/rocky/10/appstream
enabled=1
gpgcheck=0

[internal-extras]
name=Internal Rocky 10 Extras
baseurl=http://192.168.10.10/rocky/10/extras
enabled=1
gpgcheck=0
EOF

# 저장소 캐시 정리 및 목록 갱신
dnf clean all
dnf repolist
```

### Private Container (Image) Registry

Kubespray가 K8s 클러스터를 배포할 때, 각 노드에서 kube-apiserver, kube-controller-manager, coredns 등 다수의 컨테이너 이미지가 필요합니다. 폐쇄망에서는 사설 레지스트리를 구축하고 필요한 이미지를 사전에 push해 두어야 합니다.

이를 위한 이미지는 사전에 Bastion을 통해서 전송받았습니다.

![ls -l /data/registry](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*h5McVNQPwxY7rtAxvRMw1A.png) *ls -l /data/registry*

```bash
# Registry 이미지 받기
podman load -i /data/registry/registry.tar

# 이미지 확인
podman images

# Registry 데이터 저장용 하위 디렉토리 생성
mkdir -p /data/registry/storage
chmod 755 /data/registry/storage

# Docker Registry 컨테이너 실행 (기본, 인증 없음)
podman run -d \
  --name local-registry \
  -p 5000:5000 \
  -v /data/registry/storage:/var/lib/registry \
  --restart=always \
  docker.io/library/registry:latest

# Registry 상태 확인
curl -s http://localhost:5000/v2/_catalog | jq
```

Docker Registry v2는 기본적으로 HTTPS를 요구합니다. 내부망에서 HTTP로 통신하려면 insecure 설정이 필요합니다. 이를 활성화하고 이미지가 업로드되는지 확인할 수 있습니다.

```bash
# 백업
cp /etc/containers/registries.conf /etc/containers/registries.bak

# Insecure registry 설정
cat <<EOF >> /etc/containers/registries.conf
[[registry]]
location = "192.168.10.10:5000"
insecure = true
EOF

# 이미지 로드, 태그, Push
podman load -i /data/registry/alpine.tar
podman tag alpine:latest 192.168.10.10:5000/alpine:1.0
podman push 192.168.10.10:5000/alpine:1.0

# 업로드 확인
curl -s 192.168.10.10:5000/v2/_catalog
# {"repositories":["alpine"]}

curl -s 192.168.10.10:5000/v2/alpine/tags/list
# {"name":"alpine","tags":["1.0"]}
```

노드에서도 위와 동일하게 수행합니다.

```bash
cp /etc/containers/registries.conf /etc/containers/registries.bak

cat <<EOF >> /etc/containers/registries.conf
[[registry]]
location = "192.168.10.10:5000"
insecure = true
EOF

# 이미지 pull 테스트
podman pull 192.168.10.10:5000/alpine:1.0
```

### Private PyPI(Python Package Index) Mirror

Kubespray는 Ansible 기반으로 동작하며, jmespath, netaddr 등 Python 의존성 패키지를 필요로 합니다. 폐쇄망에서 pip install이 외부 PyPI에 접근할 수 없으므로, Admin에 사설 PyPI 미러를 구축합니다.

devpi는 사설 PyPI 미러 서버로, 폐쇄망에서 패키지를 다운로드받을 수 있습니다.

```bash
# devpi-server, devpi-client만 설치 (devpi-web 제외)
pip install --no-index --find-links=/data/pypi devpi-server devpi-client

# devpi 데이터 디렉토리 초기화
devpi-init --serverdir /data/devpi_data

# devpi-server 백그라운드 실행
nohup devpi-server \
  --serverdir /data/devpi_data \
  --host 0.0.0.0 \
  --port 3141 \
  > /var/log/devpi.log 2>&1 &

# devpi-server 실행 확인
ss -tnlp | grep devpi-server
```

devpi 서버를 지정하고, 관리자로 로그인을 합니다. 이후 k8s-node들이 패키지들을 다운로드할 수 있도록 /data/pypi/에 있는 패키지들을 devpi 인덱스로 업로드합니다.

```bash
# devpi 인덱스 설정
devpi use http://192.168.10.10:3141
devpi login root --password ""

# 프로젝트 인덱스 생성 (root/pypi를 상속)
devpi index -c prod bases=root/pypi
devpi use root/prod

# 패키지 업로드
devpi upload /data/pypi/*

# 업로드된 패키지 확인
devpi list
```

k8s-node들에서 pip가 어디에서 패키지를 다운로드할지 설정합니다.

```bash
cat <<EOF > /etc/pip.conf
[global]
index-url = http://192.168.10.10:3141/root/prod/+simple
trusted-host = 192.168.10.10
EOF

# 패키지 설치
pip install netaddr
```

> +simple 경로는 PEP 503(Simple Repository API)에 따른 pip 전용 인덱스 엔드포인트입니다.

### Cleanup

이후 실습을 위해서 admin에서 아래와 같이 설정을 초기화 합니다.

```bash
# nginx 서비스 중지 및 삭제
systemctl disable --now nginx
dnf remove -y nginx

# Registry 컨테이너 중지 및 삭제
podman rm -f local-registry

# 백업 파일로 원복
mv /etc/containers/registries.bak /etc/containers/registries.conf

# devpi-server 프로세스 종료
pkill -f "devpi-server --serverdir /data/devpi_data"
```

k8s-node들에서도 초기화를 수행합니다.

```bash
# registries.conf 원복
mv /etc/containers/registries.bak /etc/containers/registries.conf

# pip.conf 삭제
rm -rf /etc/pip.conf
```
