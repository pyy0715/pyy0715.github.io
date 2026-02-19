---
date: 2026-02-15 00:00:00
layout: post
title: "Bootstrapping Air-gapped Kubernetes clusters — 1"
type: Guide
math: true

category: DevOps
tags:
  - Kubernetes
  - DevOps
  - Kubespray
author: pyy0715
---

> Cloudnet@K8S-Deploy Week6

## Overview

인터넷이 차단된 Air-Gap 환경에서 Kubernetes 클러스터를 구축하려면 일반적인 설치 방식과는 전혀 다른 접근이 필요합니다. 일반적인 환경에서는 각 노드가 필요한 바이너리와 컨테이너 이미지를 인터넷에서 직접 다운로드하지만, Air-Gap 환경에서는 이것이 불가능합니다. 따라서 모든 리소스를 외부에서 미리 확보한 뒤, 이를 내부망으로 이관하여 설치를 진행해야 합니다.

![Air-Gap Environment Overview](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*H5YLrHMx3K2EtAbj-DTpAA.png)

이 글에서는 외부 네트워크에 접근할 수 있는 Bastion 서버에서 Kubernetes 클러스터 구축에 필요한 모든 리소스를 수집하고, 이를 내부 Admin 서버로 전송한 뒤, Air-Gap 환경 내부에서 클러스터를 구축하는 전체 과정을 다룹니다. 이를 위해 Kubespray 커뮤니티에서 제공하는 [kubespray-offline 프로젝트](https://github.com/kubespray-offline/kubespray-offline)를 활용하여 다운로드, 패키징, 이관, 배포 과정을 자동화합니다.

## Air-Gap Environment

Air-Gap 환경의 핵심 제약은 인터넷 접근이 가능한 곳과 Kubernetes 클러스터가 물리적으로 격리되어 있다는 점입니다. 이 제약을 해결하기 위해 세 가지 역할로 네트워크를 분리합니다.

- Bastion은 외부 네트워크와 내부 네트워크 모두에 접근할 수 있어 리소스를 수집하는 역할을 담당합니다.
- Admin은 내부망에서만 존재하면서 HTTP 서버와 컨테이너 레지스트리를 운영하여 K8s 노드들에게 리소스를 제공합니다.
- K8s 노드들은 완전히 격리된 내부망에서 실제 Kubernetes 워크로드를 실행합니다.

인터넷에서 리소스를 다운로드하려면 외부 접근이 필요하지만, Kubernetes 클러스터는 보안상 격리된 내부망에 위치해야 합니다. 이 두 가지 요구사항을 동일한 머신에서 충족할 수 없으므로, 다운로드를 담당하는 Bastion과 배포를 담당하는 Admin으로 역할을 분리합니다.

Kubernetes 구성 요소는 두 가지 형태로 배포됩니다.

1. Binary(kubelet, kubeadm, kubectl) → HTTP Server에서 다운로드
2. Container Images (kube-apiserver, coredns) → Registry에서 pull

인터넷이 차단된 환경에서는 이 두 가지 모두 내부에서 제공되어야 합니다. Admin 서버가 이 역할을 수행합니다.

## Kubespray-Offline Architecture

수동으로 Air-Gap 환경을 구축하려면 Kubernetes 바이너리를 버전별, 아키텍처별로 다운로드하고, 수십 개의 컨테이너 이미지를 pull해서 save해야 하며, OS 패키지를 reposync로 동기화하고, Python 패키지를 pip download로 확보해야 합니다. 이 모든 과정을 수동으로 수행하면 누락이나 버전 불일치 문제가 발생하기 쉽습니다.

이를 위해 지난 글에서는 Bastion, Admin, K8s Node 간의 네트워크 격리 환경을 구축하고, 수동으로 Registry, YUM Mirror, PyPI Mirror를 직접 구축하였습니다.

하지만 kubespray-offline 프로젝트는 이러한 과정을 자동화하는 스크립트 체인을 제공합니다. donwoload-all.sh가 시작점이 되어 precheck.sh, prepare-pkgs.sh, get-kubespray.sh, download-kubespray-files.sh, pypi-mirror.sh, create-repo.sh 등의 하위 스크립트를 순차적으로 호출합니다. 각 스크립트는 특정 유형의 리소스를 담당하여 최종적으로 outputs/ 디렉토리에 모든 오프라인 설치 자원을 통합합니다.

마지막 copy-target-scirpts.sh 스크립트는 Admin 서버에서 실행할 container.sh, start-nginx.sh 같은 스크립트들을 outputs/ 디렉토리로 복사합니다. 덕분에 outputs/ 디렉토리만 전송하면 Admin 서버에서 필요한 실행 스크립트까지 모두 포함됩니다.

{% raw %}
```
download-all.sh (시작점)
    │
    ├── precheck.sh          # Podman, SELinux 검증
    ├── prepare-pkgs.sh      # 시스템 패키지 설치
    ├── prepare-py.sh        # Python venv 생성
    ├── get-kubespray.sh     # Kubespray 다운로드
    ├── pypi-mirror.sh       # PyPI 패키지 미러링
    ├── download-kubespray-files.sh  # 바이너리, 이미지 다운로드
    ├── download-additional-containers.sh  # 추가 이미지
    ├── create-repo.sh       # RPM/DEB 저장소 생성
    └── copy-target-scripts.sh  # Admin용 스크립트 복사
```
{% endraw %}

### Config

config.sh는 모든 하위 스크립트가 참조하는 전역 변수 정의 파일로, KUBESPRAY_VERSION, RUNC_VERSION, CONTAINERD_VERSION, REGISTRY_PORT 등의 버전 정보를 중앙에서 관리합니다.

{% raw %}
```bash
#!/bin/bash

source ./target-scripts/config.sh

# container runtime for preparation node
docker=${docker:-podman}
#docker=${docker:-docker}
#docker=${docker:-/usr/local/bin/nerdctl}

# Run ansible in container?
ansible_in_container=${ansible_in_container:-false}
```
{% endraw %}

### Kubespray files

download-kubespray-files.sh는 Kubespray의 공식 오프라인 스크립트인 contrib/offline/generate_list.sh를 실행하여 다운로드 대상 목록을 생성합니다. 이 스크립트는 Kubespray의 Ansible 변수를 읽어 현재 배포 설정에서 실제로 사용될 파일만 목록에 포함시킵니다.

{% raw %}
```bash
generate_list() {
    LANG=C /bin/bash ${KUBESPRAY_DIR}/contrib/offline/generate_list.sh
}

generate_list
cp ${KUBESPRAY_DIR}/contrib/offline/temp/files.list $FILES_DIR/
cp ${KUBESPRAY_DIR}/contrib/offline/temp/images.list $IMAGES_DIR/
```
{% endraw %}

generate_list.sh는 내부적으로 Ansible playbook contrib/offline/generate_list.yml을 실행하며, 이 playbook은 roles/download/defaults/main.yml과 roles/kubespray-defaults/defaults/main.yml의 변수를 읽어 URL 템플릿을 실제 다운로드 URL로 변환합니다. 예를 들어, kube_version: v1.34.3 변수는 다음 URL로 확장됩니다

- [https://dl.k8s.io/release/v1.34.3/bin/linux/arm64/kubelet](https://dl.k8s.io/release/v1.34.3/bin/linux/arm64/kubelet)
- [https://dl.k8s.io/release/v1.34.3/bin/linux/arm64/kubectl](https://dl.k8s.io/release/v1.34.3/bin/linux/arm64/kubectl)
- [https://dl.k8s.io/release/v1.34.3/bin/linux/arm64/kubeadm](https://dl.k8s.io/release/v1.34.3/bin/linux/arm64/kubeadm)

이 목록은 선택된 CNI 플러그인(Calico, Cilium, Flannel 등)과 Ingress Controller(nginx-ingress, MetalLB 등)에 따라 동적으로 변경됩니다. 목록 생성 후, download-kubespray-files.sh는 get_url() 함수로 각 파일을 다운로드하며, decide_relative_dir() 함수로 저장 경로를 결정합니다.

예를 들어 kubectl, kubeadm, kubelet은 kubernetes/v1.34.3/디렉토리에, etcd는 kubernetes/etcd/에, CNI 플러그인은 kubernetes/cni/에 저장됩니다.

{% raw %}
```bash
decide_relative_dir() {
    local url=$1
    local rdir=$url
    rdir=$(echo $rdir | sed "s@.*/\(v[0-9.]*\)/.*/kube\(adm\|ctl\|let\)@kubernetes/\1@g")
    rdir=$(echo $rdir | sed "s@.*/etcd-.*.tar.gz@kubernetes/etcd@")
    rdir=$(echo $rdir | sed "s@.*/cni-plugins.*.tgz@kubernetes/cni@")
    echo $rdir
}
```
{% endraw %}

### Download image

컨테이너 이미지는 scripts/images.sh의 get_image() 함수가 처리합니다. podman이나 docker를 사용하여 이미지를 pull한 뒤 tar 형태로 저장하고 gzip으로 압축합니다. 이미지 이름에 포함된 슬래시와 콜론을 언더스코어와 대시로 변환하여 파일명을 만듭니다.

{% raw %}
```bash
get_image() {
    image=$1
    tarname="$(echo ${image} | sed s@"/"@"_"@g | sed s/":"/"-"/g)".tar
    zipname="$(echo ${image} | sed s@"/"@"_"@g | sed s/":"/"-"/g)".tar.gz

    $docker pull $image
    $docker save -o $IMAGES_DIR/$tarname $image
    gzip -v $IMAGES_DIR/$tarname
}
```
{% endraw %}

**Generate RPM Repository**

create-repo.sh는 OS 패키지 저장소를 생성합니다. RHEL/Rocky Linux의 경우 pkiglist/rhel 디렉토리의 패키지 목록을 읽어 dnf download로 의존성을 포함한 모든 RPM을 다운로드합니다. 그 후 createrepo명령으로 로컬 저장소 메타데이터를 생성합니다.

{% raw %}
```bash
PKGS=$(cat pkglist/rhel/*.txt pkglist/rhel/${VERSION_MAJOR}/*.txt | grep -v "^#" | sort | uniq)

sudo dnf download --resolve --alldeps --downloaddir $CACHEDIR $PKGS
createrepo $RPMDIR
```
{% endraw %}

## Setup

Bastion 서버는 인터넷에 접근할 수 있는 유일한 서버입니다. 따라서 Bastion에서 아래와 같이 저장소를 Clone하고 스크립트를 실행합니다.

{% raw %}
```bash
# git clone
git clone https://github.com/kubespray-offline/kubespray-offline
cd kubespray-offline/

# install
./download-all.sh
```
{% endraw %}

download-all.sh 실행 완료 후, outputs/디렉터리는 다음 구조를 가집니다.

![tree outputs directory](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*O7D-7-Iv3dSP5xE2ERoNbg.png) *tree /root/kubespray-offline/outputs/ -L 1*

- **files**: Kubespray가 K8s Node에 설치할 바이너리 파일 (kubectl, kubeadm, containerd, runc, CNI plugins 등)
- **images**: Kubernetes 컴포넌트 컨테이너 이미지 (kube-apiserver, etcd, coredns 등)
- **playbook**: 노드들에 offline repo 설정을 위한 playbook/role
- **pypi**: Ansible 의존성 Python 패키지 (jinja2, netaddr, jmespath 등)
- **rpm**: K8s Node에서 필요한 시스템 패키지

### Transfer

Kubespray는 네트워크 상황에 따라 리소스를 내려받는 세 가지 전략을 지원합니다.

1. **Default Mode** ( download_run_once: False): 각 K8s 노드가 인터넷을 통해 직접 바이너리와 이미지를 다운로드합니다.
2. **Pull Once, Push Many** (download_run_once: True): 특정 노드(주로 첫 번째 Control Plane)가 모든 리소스를 다운로드하고, 다른 노드들로 전파(Distribution)합니다.
3. **Localhost Delegation** ( download_localhost: True): Ansible을 실행하는 로컬 호스트가 모든 리소스를 다운로드한 후, 이를 클러스터 노드로 복사합니다.

kubespray-offline아키텍처는 이 Localhost Delegation 모델을 확장하여 Bastion이 다운로드한 리소스를 rsync를 통해 Admin 서버로 이관하면, Admin 서버가 HTTP 서버 및 레지스트리 역할을 수행하여 클러스터 노드들에 리소스를 제공합니다

따라서 먼저 다운로드 한 리소스를 rsync를 사용해서 Admin서버로 전송합니다. 또한 offline.yml파일은 kubespray-offline저장소의 루트 디렉토리에 위치하므로 outputs/에 포함되지 않습니다. offline.yml은 별도로 scp로 전송합니다.

{% raw %}
```bash
ssh root@admin 'mkdir -p /root/kubespray-offline/outputs'
rsync -avz --progress outputs/ root@admin:/root/kubespray-offline/outputs/
scp /root/kubespray-offline/offline.yml root@admin:/root/kubespray-offline/
```
{% endraw %}

### Admin

Admin 서버는 Air-Gap 환경 내부의 인프라 허브 역할을 합니다. 외부 네트워크가 차단된 K8s 노드들이 필요로 하는 모든 리소스를 제공하는 단일 공급원입니다. 이 역할을 수행하기 위해 Admin은 컨테이너 런타임, HTTP 서버, 컨테이너 레지스트리를 실행해야 합니다.

setup-all.sh를 실행하면 모든 설정이 한 번에 완료되지만 각 단계가 무엇을 하는지 이해하기 위해 하나씩 실행하도록 합니다.

### Containerd Installation

setup-container.sh는 containerd, runc, nerdctl, CNI plugins를 로컬 파일에서 설치하고, registry와 nginx 이미지를 로드합니다.

{% raw %}
```bash
cd /root/kubespray-offline/outputs

# containerd, runc, nerdctl, CNI 설치
./setup-container.sh

# 설치된 바이너리 파일 및 버전 확인
which runc && runc --version
which containerd && containerd --version
which nerdctl && nerdctl --version
```
{% endraw %}

아래와 같이 이미지가 로드된 것을 확인할 수 있습니다.

![nerdctl images](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*cJFbIt2wHIPOS3mrq3LlQw.png) *nerdctl images*

### HTTP File Server

start-nginx.sh는 Nginx 컨테이너를 실행하여 outputs/ 디렉터리를 HTTP로 제공합니다.

{% raw %}
```bash
cp nginx-default.conf nginx-default.bak

# nginx conf 파일 수정
cat << 'EOF' > nginx-default.conf
server {
    listen       80;
    listen  [::]:80;
    server_name  localhost;

    location / {
        root   /usr/share/nginx/html;
        autoindex on;                 # 디렉터리 목록 표시
        autoindex_exact_size off;     # 파일 크기 읽기 쉽게
        autoindex_localtime on;       # 서버 로컬 타임
    }

    error_page   500 502 503 504  /50x.html;
    location = /50x.html {
        root   /usr/share/nginx/html;
    }

    sendfile off;
}
EOF

# Nginx 컨테이너 시작
./start-nginx.sh
```
{% endraw %}

### YUM/PyPI Repository Configuration

setup-offline.sh는 외부 YUM 저장소를 비활성화하고, Nginx를 통해 제공되는 로컬 저장소만 사용하도록 설정합니다.

{% raw %}
```bash
./setup-offline.sh

# offline repo 확인
dnf clean all
dnf repolist
```
{% endraw %}

![offline.repo configuration](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*wXHophLMHQLN4N0D7omfWA.png) *cat /etc/yum.repos.d/offline.repo*

### Python Installation Verification

setup-py.sh는 로컬 YUM 저장소에서 Python을 설치하여 오프라인 저장소 동작을 검증합니다.

RHEL 10의 경우 Python 3.12가 기본이며, pyver.sh가 OS 버전에 따라 적절한 Python 버전을 자동 선택합니다.

{% raw %}
```bash
./setup-py.sh
```
{% endraw %}

### Private Container Registry

start-registry.sh는 Docker Registry v2 컨테이너를 실행하여 포트 35000에서 서비스합니다.

{% raw %}
```bash
./start-registry.sh
```
{% endraw %}

아래와 같이 컨테이너가 작동중인 것을 확인할 수 있습니다.

![nerdctl ps](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*v6LmDN-iVnSVSnuotfKriQ.png) *nerdctl ps*

### Image Upload

Bastion에서 다운로드한 컨테이너 이미지는 원본 레지스트리 주소를 태그로 가집니다. 예를 들어 registry.k8s.io/coredns/coredns:v1.12.0 형태입니다. Admin 레지스트리에 업로드하려면 이를 내부 주소인 192.168.10.10:35000/coredns/coredns:v1.12.0 형태로 변경해야 합니다.

load-push-all-images.sh는 images/ 디렉터리의 모든 .tar.gz 파일을 nerdctl load로 압축 해제한 후, Registry에 태깅하여 push합니다.

{% raw %}
```bash
push_images() {
    images=$(cat $BASEDIR/images/*.list)
    for image in $images; do
        # 원본 레지스트리 접두사 제거
        newImage=$image
        for repo in registry.k8s.io k8s.gcr.io gcr.io ghcr.io docker.io quay.io; do
            newImage=$(echo ${newImage} | sed s@^${repo}/@@)
        done

        # 내부 레지스트리 접두사 추가
        newImage=${LOCAL_REGISTRY}/${newImage}

        # 태그 변경 후 push
        $NERDCTL tag ${image} ${newImage}
        $NERDCTL push ${newImage}
    done
}
```
{% endraw %}

ARM64 환경에서는 AMD64 이미지가 포함된 tarball 로드 시 오류가 발생할 수 있으므로 nerdctl load명령에 —all-platforms플래그를 추가해야 합니다.

따라서 load-push-all-images.sh 아래와 같이 수정합니다.

![load-push-all-images.sh modification](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*6Q_FU3wFlaWudyhUphz-9g.png)

이후 스크립트를 실행합니다.

{% raw %}
```bash
./load-push-all-images.sh
```
{% endraw %}

저장소에 생성된 이미지 디렉토리를 확인할 수 있습니다.

![registry directory structure](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*sLw2jVmsx0SIAH77p5dUGA.png) *tree /var/lib/registry/ -L 5*

### Kubespray Extraction

extract-kubespray.sh는 Kubespray tarball을 압축 해제하고 패치를 적용합니다.

{% raw %}
```bash
./extract-kubespray.sh
```
{% endraw %}

압축 해제된 디렉터리를 아래와 같이 확인할 수 있습니다.

![kubespray directory structure](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Lt7Wf5ETXqtaJxawI6OCSQ.png) *tree kubespray-2.30.0/ -L 1*

## Install Kubespray

지금까지 구축한 인프라를 Kubespray가 인식하도록 설정해야 합니다. Kubespray는 기본적으로 인터넷에서 리소스를 다운로드하도록 설계되어 있으므로, 이를 내부 리소스를 사용하도록 재구성해야 합니다.

Ansible은 변수 우선순위 규칙을 가집니다. 기본값(defaults)보다 그룹 변수(group_vars)가 우선하고, 그룹 변수보다 호스트 변수가 우선합니다. Kubespray의 기본 다운로드 URL은 roles/download/defaults/main.yml에 정의되어 있는데, offline.yml을 inventory/mycluster/group_vars/all/offline.yml에 배치하면 그룹 변수가 기본값을 덮어씁니다.

### offline.yml

offline.yml은 **폐쇄망 클러스터의 리소스 위치를 선언**하는 파일로 Kubespray의 태스크가 어디서 파일을 다운로드하고, 어디서 이미지를 pull할지 결정합니다.

{% raw %}
```yaml
# kubespray-offline/offline.yml

http_server: "http://YOUR_HOST"
registry_host: "YOUR_HOST:35000"

# Containerd가 내부 레지스트리를 신뢰하도록 설정
containerd_registries_mirrors:
  - prefix: "{{ registry_host }}"
    mirrors:
      - host: "http://{{ registry_host }}"
        capabilities: ["pull", "resolve"]
        skip_verify: true

# 리소스 저장소 위치
files_repo: "{{ http_server }}/files"
yum_repo: "{{ http_server }}/rpms"
ubuntu_repo: "{{ http_server }}/debs"

# 모든 이미지를 내부 레지스트리에서 가져오도록 리다이렉트
kube_image_repo: "{{ registry_host }}"
gcr_image_repo: "{{ registry_host }}"
docker_image_repo: "{{ registry_host }}"
quay_image_repo: "{{ registry_host }}"
github_image_repo: "{{ registry_host }}"
```
{% endraw %}

내부 엔드포인트를 아래와 같이 정의하면 Kubespray는 이제 인터넷이 아닌 내부 http_server에서 파일을 찾습니다.

{% raw %}
```yaml
http_server: "http://192.168.10.10"
registry_host: "192.168.10.10:35000"
```
{% endraw %}

이는 아래 네가지 영역을 제어합니다.

**HTTP File Server Location (바이너리 다운로드)**

노드에 설치되는 kubelet, kubectl, kubeadm, etcd와 같은 구성 요소들은 모두 단일 바이너리 형태로 배포됩니다. 일반적인 인터넷 환경에서는 이 파일들을 Kubernetes 공식 배포 서버나 GitHub 릴리스 페이지에서 직접 내려받습니다. 그러나 폐쇄망에서는 이러한 URL에 접근할 수 없으므로, 사전에 동일한 버전의 바이너리들을 내부 HTTP 서버에 저장해 두고 모든 설치 스크립트가 그 서버만 바라보도록 구성합니다.

{% raw %}
```yaml
files_repo: "{{ http_server }}/files"
```
{% endraw %}

**OS Package Repository**

Kubernetes 는 컨테이너 오케스트레이션 시스템이지만, 실제로는 리눅스 커널 기능과 여러 시스템 도구에 크게 의존합니다. 예를 들어 네트워크 포워딩을 위한 socat, 커넥션 트래킹을 위한 conntrack, 네트워크 정책에 필요한 ipset 등은 모두 OS 패키지 관리자를 통해 설치됩니다. 인터넷이 열려 있을 때는 배포판의 공식 미러 서버에서 자동으로 내려받지만, 폐쇄망에서는 이 역시 불가능하므로 내부에 YUM/DNF 저장소를 구성해 필요한 RPM 패키지들을 미리 동기화해 두어야 합니다.

{% raw %}
```yaml
yum_repo: "{{ http_server }}/rpms"
```
{% endraw %}

**Containerd Registry Configuration**

쿠버네티스 노드에서 실제로 이미지를 내려받는 주체는 kubelet이 아니라 그 아래의 컨테이너 런타임, containerd 입니다. 이 런타임은 기본적으로 HTTPS와 TLS 인증서 검증을 전제로 동작하도록 설계되어 있습니다. 그러나 폐쇄망 내부 레지스트리는 자체 서명 인증서를 사용하거나, 경우에 따라 HTTP만 허용하기도 합니다. 따라서 containerd 설정에서 어떤 레지스트리를 신뢰할지, TLS 검증을 생략할지, HTTP 접근을 허용할지를 명시적으로 정의해야 합니다.

{% raw %}
```yaml
containerd_registries_mirrors:
  - prefix: "{{ registry_host }}"
    mirrors:
      - host: "http://{{ registry_host }}"
        skip_verify: true
```
{% endraw %}

> For air-gapped clusters, you may need to configure insecure registries if setting up TLS is not feasible.

이 구성이 완료되면 Kubespray의 Ansible Playbook 실행 시 외부 네트워크 호출 없이 모든 리소스가 내부망에서 해결됩니다.

offline.yml을 inventory/mycluster/group_vars/all/offline.yml에 배치하면, Ansible은 이 변수를 모든 Role에 주입합니다.

{% raw %}
```bash
cd /root/kubespray-offline/outputs/kubespray-2.30.0

# inventory 디렉토리 생성
cp -r inventory/sample inventory/mycluster

# offline.yml 배치
cp ../../offline.yml inventory/mycluster/group_vars/all/offline.yml

# Admin 서버 IP로 수정
sed -i "s/YOUR_HOST/192.168.10.10/g" inventory/mycluster/group_vars/all/offline.yml

# etcd download url 수정
sed -i 's/linux-amd64/linux-{{ image_arch }}/' inventory/mycluster/group_vars/all/offline.yml
```
{% endraw %}

> etcd download url의 경우, 공식 Kubespray 문서나 예제 파일에서 아키텍처가 amd64로 하드코딩되어 있기 때문에 image_arch 변수를 사용하여 아키텍처를 동적으로 반영하도록 수정해야 합니다.

inventory 파일은 아래와 같이 작성합니다.

{% raw %}
```bash
cat <<EOF > inventory/mycluster/inventory.ini
[kube_control_plane]
k8s-node1 ansible_host=192.168.10.11 ip=192.168.10.11 etcd_member_name=etcd1

[etcd:children]
kube_control_plane

[kube_node]
k8s-node2 ansible_host=192.168.10.12 ip=192.168.10.12
EOF
```
{% endraw %}

아래와 같이 ping모듈을 실행하여 연결을 미리 확인합니다.

![ansible ping test](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*NHNl0kIW00he-VcXxhvvzA.png) *ansible -i inventory/mycluster/inventory.ini all -m ping*

### Execute Playbook

K8s 노드들이 패키지를 설치할 때 로컬 저장소를 사용하도록 구성해야 합니다. offline-repo.yml플레이북을 실행하면 각 노드에 /etc/yum.repos.d/offline.repo가 생성됩니다. 이 작업이 cluster.yml보다 먼저 수행되어야 합니다.

{% raw %}
```bash
# 플레이북 복사 및 실행
mkdir offline-repo
cp -r ../playbook/ offline-repo/
ansible-playbook -i inventory/mycluster/inventory.ini offline-repo/playbook/offline-repo.yml
```
{% endraw %}

노드가 인터넷 연결을 시도하다가 타임아웃으로 설치가 실패하는 것을 방지하기 위해, 기존의 공개 저장소를 백업 후 비활성화합니다.

{% raw %}
```bash
# 기존 repo 제거 (미실행 시 fail 발생 가능)
for i in rocky-addons rocky-devel rocky-extras rocky; do
  ssh k8s-node1 "mv /etc/yum.repos.d/$i.repo /etc/yum.repos.d/$i.repo.original"
  ssh k8s-node2 "mv /etc/yum.repos.d/$i.repo /etc/yum.repos.d/$i.repo.original"
done
```
{% endraw %}

기본 설정값에서 실습 환경에 맞게 변수를 조정합니다.

{% raw %}
```bash
# kubectl을 Admin 서버에도 설치
echo "kubectl_localhost: true" >> inventory/mycluster/group_vars/k8s_cluster/k8s-cluster.yml

# 소유자 변경 (실습 환경)
sed -i 's|kube_owner: kube|kube_owner: root|g' inventory/mycluster/group_vars/k8s_cluster/k8s-cluster.yml

# Flannel 사용
sed -i 's|kube_network_plugin: calico|kube_network_plugin: flannel|g' inventory/mycluster/group_vars/k8s_cluster/k8s-cluster.yml

# iptables 모드 사용
sed -i 's|kube_proxy_mode: ipvs|kube_proxy_mode: iptables|g' inventory/mycluster/group_vars/k8s_cluster/k8s-cluster.yml

# NodeLocal DNS 비활성화
sed -i 's|enable_nodelocaldns: true|enable_nodelocaldns: false|g' inventory/mycluster/group_vars/k8s_cluster/k8s-cluster.yml

# Flannel NIC 지정
echo "flannel_interface: enp0s9" >> inventory/mycluster/group_vars/k8s_cluster/k8s-net-flannel.yml

# Helm 활성화
sed -i 's|helm_enabled: false|helm_enabled: true|g' inventory/mycluster/group_vars/k8s_cluster/addons.yml

# Metrics Server 활성화
sed -i 's|metrics_server_enabled: false|metrics_server_enabled: true|g' inventory/mycluster/group_vars/k8s_cluster/addons.yml
echo "metrics_server_requests_cpu: 25m" >> inventory/mycluster/group_vars/k8s_cluster/addons.yml
```
{% endraw %}

모든 준비가 완료되면 cluster.yml을 실행합니다. -e kube_version으로 명시적으로 버전을 지정하면 Kubespray가 의도치 않게 다른 버전을 설치하는 것을 방지할 수 있고, offline.yml의 URL 템플릿이 올바르게 확장됩니다.

{% raw %}
```bash
ansible-playbook -i inventory/mycluster/inventory.ini -v cluster.yml -e kube_version="1.34.3"
```
{% endraw %}

## Post-Deployment

group_vars에서 kubectl_localhost:true로 설정했다면, Kubespray는 설치 과정에서 사용된 kubectl 바이너리를 Ansible 실행 디렉터리의 artifacts/경로에 자동으로 다운로드합니다.

{% raw %}
```bash
# 바이너리 확인
ls -l inventory/mycluster/artifacts/kubectl

# 시스템 경로로 이동 및 권한 부여
cp inventory/mycluster/artifacts/kubectl /usr/local/bin/
chmod +x /usr/local/bin/kubectl

# 버전 확인
kubectl version --client=true
```
{% endraw %}

기본적으로 Control Plane 노드의 kubeconfig는 localhost(127.0.0.1)를 API 서버 엔드포인트로 설정합니다. Admin 서버와 Control Plane이 다른 호스트인 경우, 원격 접속을 위해 이 IP 주소를 실제 노드의 IP(192.168.10.11)로 변경해야 합니다.

{% raw %}
```bash
# config 파일 복사
mkdir -p /root/.kube
scp k8s-node1:/root/.kube/config /root/.kube/

# 127.0.0.1을 Control Plane의 실제 IP로 변경
sed -i 's/127.0.0.1/192.168.10.11/g' /root/.kube/config

# 연결 확인
kubectl get nodes
```
{% endraw %}

이후, 모든 파드의 이미지가 내부 레지스트리(192.168.10.10:35000)에서 Pull 되었는지 확인합니다.

{% raw %}
```bash
kubectl get deploy,sts,ds -n kube-system -owide
```
{% endraw %}

![kubectl get deploy output](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*rk-dskFq1iY-tZ7ELaUHgA.png)
