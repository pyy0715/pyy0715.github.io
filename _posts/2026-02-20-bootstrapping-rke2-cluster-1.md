---
date: 2026-02-20 00:00:00
layout: post
title: "Bootstrapping RKE2 Cluster — 1"
type: Guide
math: true

category: DevOps
tags:
  - Kubernetes
  - RKE2
  - DevOps
author: pyy0715
---

## Lab Environment

RKE2 구성을 위해 Rocky Linux 9 VM 2대를 Vagrant로 배포합니다.

```bash
mkdir k8s-rke2
cd k8s-rke2

curl -O https://raw.githubusercontent.com/gasida/vagrant-lab/refs/heads/main/k8s-rke2/Vagrantfile
curl -O https://raw.githubusercontent.com/gasida/vagrant-lab/refs/heads/main/k8s-rke2/init_cfg.sh

vagrant up
vagrant status
```

## RKE2 Installation

RKE2는 설치 환경의 운영체제 및 네트워크 상태에 따라 최적의 배포 방식을 결정할 수 있는 설치 스크립트를 제공합니다. Rocky Linux 9와 같은 RPM 기반 시스템에서는 스크립트가 내부적으로 YUM/DNF 리포지토리를 구성하여 패키지 형태로 설치를 진행합니다.

### Server Node Installation

설치 스크립트는 실행 시 환경변수(`INSTALL_RKE2_CHANNEL`, `INSTALL_RKE2_METHOD` 등)를 읽어들여 동작 방식을 결정합니다.

```bash
curl -sfL https://get.rke2.io --output install.sh
chmod +x install.sh
INSTALL_RKE2_CHANNEL=v1.33 ./install.sh
```

`/etc/os-release`를 파싱해 OS 릴리즈 정보를 기반으로 `/etc/yum.repos.d/rancher-rke2.repo` 파일을 자동 생성하며, 지정된 버전(v1.33)의 `rke2-server` 패키지와 SELinux 정책을 설치합니다.

### Control Plane Configuration

RKE2 서비스가 구동되기 전, 클러스터의 네트워크 아키텍처와 Add-on의 활성화 여부를 결정해야 합니다. `/etc/rancher/rke2/config.yaml` 파일은 Control Plane의 기본 동작을 제어하는 설정 파일입니다. 이 단계에서 CNI(Container Network Interface) 플러그인을 선택하고 불필요한 기본 컨트롤러들을 비활성화합니다.

{% raw %}
```yaml
cat << EOF > /etc/rancher/rke2/config.yaml
write-kubeconfig-mode: "0644"

debug: true

cni: canal

bind-address: 192.168.10.11
advertise-address: 192.168.10.11
node-ip: 192.168.10.11

disable-cloud-controller: true

disable:
  - servicelb
  - rke2-coredns-autoscaler
  - rke2-ingress-nginx
  - rke2-snapshot-controller
  - rke2-snapshot-controller-crd
  - rke2-snapshot-validation-webhook
EOF
```
{% endraw %}

설정 파일에서 `cni: canal`을 지정하고, `disable` 블록을 통해 `servicelb`, `rke2-ingress-nginx` 등을 설치 대상에서 제외했습니다. 이는 RKE2가 기본적으로 제공하는 패키징된 리소스를 로드하지 않도록 kubelet 및 Helm 컨트롤러에 지시하는 역할을 합니다.

### Adds-on

RKE2는 CNI, CoreDNS, Metrics Server 등의 핵심 시스템 컴포넌트를 Helm Chart 형태로 관리합니다. 사용자가 이들의 기본 설정을 덮어쓰기 위해서는 RKE2 전용 커스텀 리소스인 `HelmChartConfig`를 활용해야 합니다. 클러스터가 시작될 때, Helm 컨트롤러가 이 파일을 읽어 원본 Helm Chart의 `values.yaml`을 재정의합니다.

{% raw %}
```yaml
# 디렉토리 생성
mkdir -p /var/lib/rancher/rke2/server/manifests/

# Canal
cat << EOF > /var/lib/rancher/rke2/server/manifests/rke2-canal-config.yaml
apiVersion: helm.cattle.io/v1
kind: HelmChartConfig
metadata:
  name: rke2-canal
  namespace: kube-system
spec:
  valuesContent: |-
    flannel:
      iface: "enp0s9"
EOF

# CoreDNS
cat << EOF > /var/lib/rancher/rke2/server/manifests/rke2-coredns-config.yaml
apiVersion: helm.cattle.io/v1
kind: HelmChartConfig
metadata:
  name: rke2-coredns
  namespace: kube-system
spec:
  valuesContent: |-
    autoscaler:
      enabled: false
EOF
```
{% endraw %}

`/var/lib/rancher/rke2/server/manifests/` 디렉터리에 `rke2-canal-config.yaml`과 `rke2-coredns-config.yaml`을 생성하여, Canal CNI가 특정 네트워크 인터페이스(`enp0s9`)를 사용하도록 바인딩하고 CoreDNS의 오토스케일러를 비활성화했습니다.

클러스터 구동 후 `kubectl edit configmap`을 통해 CoreDNS 설정을 직접 수정하더라도, RKE2의 Helm 컨트롤러에 의해 원래 상태로 덮어씌워집니다. 따라서 모든 애드온 설정 변경은 반드시 `HelmChartConfig`를 통해 파일 기반으로 이루어져야만 영속성을 보장받을 수 있습니다.

### Service Initialization

모든 사전 설정이 완료되면 systemd를 통해 RKE2 서버 프로세스를 시작합니다. 이 단계에서 containerd 런타임이 구동되고, Static Pod 형태로 Control Plane 컴포넌트들이 올라오며, 지정된 Add-on들이 Helm을 통해 배포됩니다.

```bash
# 서비스 구동 및 상태 확인
systemctl enable --now rke2-server.service

# 권한 증명 복사
mkdir ~/.kube
cp /etc/rancher/rke2/rke2.yaml ~/.kube/config

# 클러스터 정상 구동 확인 (Control Plane 및 etcd 역할 확인)
kubectl get node -o wide
```

### Runtime Layout

RKE2는 호스트 OS의 패키지 의존성을 제거하기 위해 containerd, runc, kubectl 등의 바이너리를 내장하여 배포합니다. 이 바이너리들은 기본적으로 호스트의 PATH에 포함되지 않으므로, 시스템 전역에서 명령어를 사용하기 위해 심볼릭 링크를 통한 경로 설정이 필요합니다.

내장된 바이너리들은 `/var/lib/rancher/rke2/bin/`에 위치하며, crictl 이 containerd와 통신하기 위해 `crictl.yaml` 파일에서 런타임 엔드포인트를 RKE2가 사용하는 containerd 소켓 경로인 `unix:///run/k3s/containerd/containerd.sock`을 바라보도록 설정되어 있습니다.

![tree /var/lib/rancher/rke2/bin/](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*0sE_Dimw8L2iz7S5oF4fxw.png) *tree /var/lib/rancher/rke2/bin/*

```bash
ln -s /var/lib/rancher/rke2/bin/containerd /usr/local/bin/containerd
ln -s /var/lib/rancher/rke2/bin/kubectl /usr/local/bin/kubectl
ln -s /var/lib/rancher/rke2/bin/crictl /usr/local/bin/crictl
ln -s /var/lib/rancher/rke2/bin/runc /usr/local/bin/runc
ln -s /var/lib/rancher/rke2/bin/ctr /usr/local/bin/ctr
ln -s /var/lib/rancher/rke2/agent/etc/crictl.yaml /etc/crictl.yaml
```

이후 버전 확인을 통해 바이너리가 인식되는지 확인할 수 있습니다.

```bash
runc --version
containerd --version
kubectl version
```

### Directory Architecture

RKE2는 호스트 OS의 파일 시스템을 건드리지 않고, 자체 생태계를 독립적으로 유지하기 위해 상태와 구성 요소를 `/var/lib/rancher/rke2` 하위로 격리합니다. 이 디렉터리는 역할에 따라 크게 `server`, `agent`, `data` 세 가지 구조로 나뉩니다.

- `server/`: etcd 데이터베이스, Kubeconfig 자격 증명, 인증서, 그리고 애드온 배포를 위한 `manifests`가 위치합니다. 다른 노드를 클러스터에 편입시키기 위한 조인 토큰 역시 `server/node-token`에 생성됩니다.
- `agent/`: 워커 노드(kubelet)의 실행 환경입니다. containerd 런타임 환경, 파드 실행을 위한 정적 매니페스트(`pod-manifests`), Kubelet 로그 및 통신용 클라이언트 인증서가 포함됩니다.
- `data/`: 특정 RKE2 버전에 종속된 정적 데이터입니다. 핵심 바이너리와 Helm Chart 원본이 저장되며, `/var/lib/rancher/rke2/bin`은 현재 구동 중인 버전의 `data` 디렉터리로 심볼릭 링크되어 구동됩니다.

![tree /var/lib/rancher/rke2 -L 1](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*SPZDlUViIDnR2rZLhupmzQ.png) *tree /var/lib/rancher/rke2 -L 1*

![tree /var/lib/rancher/rke2/server/ -L 1](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*pvKvD_-pI1fCOvgyS2brEA.png) *tree /var/lib/rancher/rke2/server/ -L 1*

![tree /var/lib/rancher/rke2/agent/ -L 1](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*C9gzsB2TZIXlmZxZIQaDDw.png) *tree /var/lib/rancher/rke2/agent/ -L 1*

![tree /var/lib/rancher/rke2/data/ -L 3](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*aupFmrfpzWHB06qH8wX8Dg.png) *tree /var/lib/rancher/rke2/data/ -L 3*

### Helm Controller Lifecycle

RKE2는 Canal(CNI), CoreDNS, Metrics Server와 같은 주요 시스템 애드온을 외부 Helm CLI가 아닌 내장 Helm 컨트롤러를 통해 관리합니다. 이는 설치 스크립트가 직접 `helm install`을 호출하는 구조가 아니라, Kubernetes 리소스를 기반으로 한 선언적 수렴 모델로 동작합니다.

RKE2 서버는 `/var/lib/rancher/rke2/server/manifests` 디렉터리를 지속적으로 감시합니다. 이 디렉터리에 manifest 파일이나 `HelmChartConfig` 리소스 정의를 배치하면, 컨트롤러는 이를 감지하여 내부 처리 과정을 시작합니다.

파일이 처음 감지되면 컨트롤러는 해당 파일의 내용을 읽어 SHA256과 같은 해시 값을 계산합니다. 이 해시는 파일의 변경 여부를 판단하기 위한 식별자입니다. 계산된 값은 `addons.k3s.cattle.io` 리소스의 상태 필드에 기록됩니다.

![kubectl get addons.k3s.cattle.io -A](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*CH9PzGAbNh3Agois-BwtcQ.png) *kubectl get addons.k3s.cattle.io -A*

이후 서버가 재시작되거나 manifests 디렉터리를 다시 스캔할 때, 컨트롤러는 동일 파일의 해시를 다시 계산합니다. 새로 계산된 해시가 기존에 저장된 값과 다를 경우에만 변경이 발생한 것으로 간주합니다. 이때 컨트롤러는 기존 HelmChart 리소스를 갱신하거나 필요한 경우 재조정을 트리거합니다.

그 결과 Helm 컨트롤러가 이를 감지하고 `kube-system` 네임스페이스에 `helm-install-<name>` 형식의 일회성 Job을 생성하여 install 또는 upgrade를 수행합니다.

![kubectl get helmcharts.helm.cattle.io -n kube-system -o wide](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*KQDLMmUcXjYtgs7ltqoKwA.png) *kubectl get helmcharts.helm.cattle.io -n kube-system -o wide*

![kubectl get job -n kube-system](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*O_R92Q_2JqaNar_lu_2fKQ.png) *kubectl get job -n kube-system*

### Containerd Configuration

RKE2는 Kubelet과 containerd를 통합 관리하므로 컨테이너 런타임의 직접적인 설정 변경을 제한합니다.

Private Registry 인증 정보나 런타임 옵션을 추가하기 위해 `/var/lib/rancher/rke2/agent/etc/containerd/config.toml` 파일을 직접 수정할 경우, RKE2 서비스 재시작 시 해당 파일이 초기화됩니다. [공식 문서](https://docs.rke2.io/advanced#configuring-containerd)에서 권장하는 방식은 Go 텍스트 템플릿을 활용하는 것입니다.

> RKE2 v1.31.6 및 v1.32.2 이상 버전부터는 containerd 2.0이 내장되어** config version 3** 형식을 선호합니다. 따라서 최신 환경에서는 `config-v3.toml.tmpl` 파일을 사용해야 하며, containerd 1.7 이하의 레거시 환경에서만 `config.toml.tmpl`을 사용합니다.

```bash
# 1. RKE2가 생성한 기본 설정을 템플릿 파일로 복사
cp /var/lib/rancher/rke2/agent/etc/containerd/config.toml /var/lib/rancher/rke2/agent/etc/containerd/config-v3.toml.tmpl

# 2. 템플릿 파일 수정 (예: 이미지 업데이트)
sed -i 's/rancher\/mirrored-pause:3\.6/rancher\/mirrored-pause:3\.8/g' \
  /var/lib/rancher/rke2/agent/etc/containerd/config-v3.toml.tmpl

# 3. RKE2 서버 노드 서비스 재시작 (템플릿 기반으로 config.toml 재생성)
systemctl restart rke2-server

# 4. 동적으로 생성된 config.toml 파일에 변경 사항이 반영되었는지 확인
grep -n 'mirrored-pause' /var/lib/rancher/rke2/agent/etc/containerd/config.toml
```

이러한 절차를 거치면 RKE2의 관리 주기 내에서도 containerd 설정이 유지됩니다.

## Security

RKE2의 가장 큰 특징은 CIS(Center for Internet Security) 벤치마크를 기본적으로 충족하도록 설계된 보안 구성입니다.

`/var/lib/rancher/rke2/agent/pod-manifests/` 디렉터리에 위치한 Static Pod manifest들을 분석해보면,

## Agent Node

### Node Join

워커 노드(Agent)를 클러스터에 편입시키는 과정은 Control Plane에서 발급한 조인 토큰과 전용 통신 포트를 기반으로 이루어집니다.

서버 노드는 새로운 노드의 등록 요청을 수신하기 위해 `9345` 포트위에서 동작합니다. 에이전트 노드는 설치 스크립트 실행 후, 해당 서버 주소와 토큰이 명시된 `/etc/rancher/rke2/config.yaml` 파일을 구성하여 부트스트랩을 수행합니다

```bash
# [k8s-node1]

# Control Plane 노드에서 조인 토큰 확인
cat /var/lib/rancher/rke2/server/node-token

# 포트 확인
ss -tnlp | grep 9345
```

{% raw %}
```bash
# [k8s-node2]

# Run the installer
curl -sfL https://get.rke2.io | INSTALL_RKE2_TYPE="agent" INSTALL_RKE2_CHANNEL=v1.33 sh -

TOKEN="위에서 출력된 토큰 Copy"

mkdir -p /etc/rancher/rke2/
cat << EOF > /etc/rancher/rke2/config.yaml
server: https://192.168.10.11:9345
token: $TOKEN
EOF
cat /etc/rancher/rke2/config.yaml

# Enabled/Start the service
systemctl enable --now rke2-agent.service
journalctl -u rke2-agent -f
```
{% endraw %}

이후 정상적으로 조인된 것을 확인할 수 있습니다.

![kubectl get node -o wide](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ef9UqmI480Jw_-3rgqW7MQ.png) *kubectl get node -o wide*

### Agent Runtime Layout

에이전트 노드가 정상적으로 등록된 후 내부 아키텍처를 확인해 보면, Control Plane과는 다른 구성 요소 배치를 확인할 수 있습니다.

Control Plane과 달리 에이전트 노드의 `/var/lib/rancher/rke2/server` 디렉터리는 비어 있으며, `pstree`를 통해 확인된 프로세스 계층은 `rke2` 데몬이 `containerd`와 `kubelet` 하위 프로세스를 직접 관리하는 형태로 구성됩니다.

![pstree -al](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*HnJyENdCrl52j_yiCWEZdw.png) *pstree -al*

또한, 에이전트 노드 역시 `crictl` 및 `containerd` 명령어 사용을 위해 심볼릭 링크 설정이 권장됩니다.

```bash
ln -s /var/lib/rancher/rke2/bin/containerd /usr/local/bin/containerd
ln -s /var/lib/rancher/rke2/bin/crictl /usr/local/bin/crictl
ln -s /var/lib/rancher/rke2/agent/etc/crictl.yaml /etc/crictl.yaml
```

일반적인 쿠버네티스 고가용성(HA) 환경에서는 컨트롤 플레인이 여러 대 존재합니다. 이때 워커 노드들이 컨트롤 플레인과 통신하려면, 여러 서버 중 어느 곳으로 요청을 보내야 할지 결정해 주는 **외부 로드 밸런서 장비**(HAProxy, Nignx 등)이 필요합니다.

하지만 RKE2는 이러한 외부 의존성을 없애기 위해 Clinet-Side Load Balancer를 사용합니다.

에이전트 노드는 처음에 생성한`/etc/rancher/rke2/config.yaml`에 적어둔 단일 서버(`192.168.10.11`)로만 접속을 시도합니다.

접속에 성공하면, RKE2 데몬이 서버로부터 현재 살아있는 전체 Control Plane 노드의 IP 목록을 동적으로 받아와서 JSON 파일에 업데이트합니다.

![cat /var/lib/rancher/rke2/agent/etc/rke2-api-server-agent-load-balancer.json | jq](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ZGVMMRkIncQeiWffS9scUQ.png) *cat /var/lib/rancher/rke2/agent/etc/rke2-api-server-agent-load-balancer.json | jq*

이후 에이전트 노드의 kublet은 서버 노드로 직접 요청을 보내지 않고 로컬 프록시로 요청을 보냅니다.

![grep server /var/lib/rancher/rke2/agent/kubelet.kubeconfig](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*CWosBiM14_1GKtU5PwGvYw.png) *grep server /var/lib/rancher/rke2/agent/kubelet.kubeconfig*

127.0.0.1의 6443(API) 포트를 RKE2가 수신 중인지 확인할 수 있습니다.

![ss -tnlp | grep -E "127.0.0.1:6443"](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Gx0X59L5JhAMWkphMMvnHQ.png) *ss -tnlp | grep -E "127.0.0.1:6443"*

즉, 위에서 수신 중인 RKE2 로컬 프로세스는 내부적으로 `/var/lib/rancher/rke2/agent/etc/rke2-api-server-agent-load-balancer.json`의 주소 목록을 참고하여 실제 서버 노드로 트래픽을 분산 및 포워딩합니다.

이를 통해 Kubelet과 Kube-proxy가 외부의 전용 로드 밸런서(L4/L7) 장비 없이도 다수의 Control Plane 노드와 고가용성(HA) 통신을 유지할 수 있도록 돕습니다.

### Node Delete & Re-Joining

워크로드를 다른 노드로 Drain시킨 후 노드를 클러스터에서 제거합니다.

```bash
# 노드 스케줄 차단 + 파드 안전 이동 (Drain)
kubectl drain k8s-node2 --ignore-daemonsets --delete-emptydir-data

# Kubernetes 클러스터에서 노드 제거 : 파드가 다 빠진 거 확인 후
kubectl delete node k8s-node2
```

대상 에이전트 노드에서 RKE2 서비스를 중지하고 제공된 삭제 스크립트(`rke2-uninstall.sh`)를 실행합니다. 이 스크립트는 컨테이너, 네트워크 인터페이스, 관련 파일을 모두 일괄 정리합니다.

```bash
# RKE2 서비스 중지
systemctl stop rke2-agent

# RKE2 삭제 스크립트 실행
rke2-uninstall.sh
```

RKE2 에이전트를 다시 설치하고 클러스터에 Join시킵니다.

{% raw %}
```bash
[k8s-node2]

# Run the installer
curl -sfL https://get.rke2.io | INSTALL_RKE2_TYPE="agent" INSTALL_RKE2_CHANNEL=v1.33 sh -

# Configure the rke2-agent service
TOKEN=<REDACTED_TOKEN>

cat << EOF > /etc/rancher/rke2/config.yaml
server: https://192.168.10.11:9345
token: $TOKEN
EOF

# Enabled/Start the service
systemctl enable --now rke2-agent.service
```
{% endraw %}

## Certificate Lifecycle

RKE2의 내부 컴포넌트 간 통신은 모두 mTLS로 보호되며, 클라이언트 및 서버 인증서의 기본 유효 기간은 365일입니다. 만료일이 120일 이내로 도래하면 서비스 재시작 시 기존 키를 재사용하여 자동으로 만료일을 연장합니다.

새로운 키 쌍(Key Pair)을 강제로 생성하여 인증서를 전면 교체해야 할 경우, 수동 갱신(Rotation) 절차가 필요합니다. 에이전트 노드는 기존 연결이 끊어지면 서버에 재접속하여 새로운 CA를 기반으로 인증서를 자동으로 재발급받습니다.

```bash
# 인증서 유효 기간 확인
rke2 certificate check --output table

# RKE2 서비스 중지
systemctl stop rke2-server

# 갱신
rke2 certificate rotate

# 재시작
systemctl start rke2-server

# 관리자 자격 증명(kubeconfig) 수동 갱신
cp /etc/rancher/rke2/rke2.yaml ~/.kube/config
```

## Manual Version Upgrade

RKE2 업그레이드는 Version Skew Policy을 엄격히 따릅니다. 따라서 중간 마이너 버전을 건너뛰는 업그레이드는 허용되지 않으며, 서버 노드를 선행하여 업그레이드한 후 에이전트 노드를 업그레이드하는 순서 제약을 지켜야 합니다.

RKE2는 업그레이드할 대상 버전을 결정할 때 자체적인 릴리스 채널(Release Channels) 메커니즘을 사용합니다. 채널 서버 API(`https://update.rke2.io/v1-release/channels`)를 통해 버전을 동적으로 매핑하며, 인프라의 목적과 검증 수준에 따라 다음과 같은 채널을 선택할 수 있습니다.

- **stable:** 커뮤니티의 광범위한 검증을 거친 안정화 버전으로, 운영 환경에 권장됩니다. Rancher 컨트롤 플레인의 최신 릴리스와의 호환성이 보장됩니다.
- **latest:** 가장 최신 기능과 패치가 포함된 버전입니다. 새로운 기능을 조기에 테스트할 수 있다는 장점이 있으나, 아직 완벽한 커뮤니티 검증이나 Rancher 플랫폼과의 호환성이 보장되지 않는다는 운영상 제약(Trade-off)을 감수해야 합니다.
- **마이너 버전 채널 (예: v1.34):** 특정 쿠버네티스 마이너 버전에 종속된 채널입니다. 해당 채널은 `stable` 채널 승인 여부와 관계없이 해당 마이너 버전 체계 내에서 사용 가능한 가장 최신 패치(Patch) 버전을 추적합니다.

실제 업그레이드 스크립트를 실행하기 전, 다음 검증 명령어를 통해 현재 설정하려는 채널이 실제로 어떤 RKE2 릴리스 버전으로 매핑되는지 API를 호출하여 사전 확인하는 것이 안전합니다.

```bash
curl -s https://update.rke2.io/v1-release/channels | jq '.data'
```

수동 업그레이드는 대상 노드에 접속하여 공식 설치 스크립트(`install.sh`)를 환경 변수와 함께 다시 실행하는 방식입니다. 스크립트 실행 시 패키지 매니저(RPM 등)가 새로운 바이너리를 다운로드하며, 서비스 데몬을 재시작할 때 Static Pod들이 신규 버전으로 재기동됩니다.

```bash
curl -sfL https://get.rke2.io | INSTALL_RKE2_CHANNEL=v1.34 sh -
systemctl restart rke2-server
```

아래와 같이 Pod들이 모두 재시작 된 것을 확인할 수 있습니다.

![kubectl get pod -n kube-system — sort-by=.metadata.creationTimestamp | tac](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*kcxRiuJ11zcDM7wo7OXUkA.png) *kubectl get pod -n kube-system — sort-by=.metadata.creationTimestamp | tac*

완료한 후, 에이전트 노드도 업그레이드해야 합니다.

```bash
curl -sfL https://get.rke2.io | INSTALL_RKE2_TYPE=agent INSTALL_RKE2_CHANNEL=v1.34 sh -
systemctl restart rke2-agent
```

RKE2환경에서 에이전트 노드의 경우, Pod 형태로 관리되는 시스템 컴포넌트가 사실상 kube-proxy하나이기 때문에 신규로 재기동됩니다.

![kubectl get pod -n kube-system — sort-by=.metadata.creationTimestamp | tac](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*B9ZsH2bbUv9ukA9aWh-IJQ.png) *kubectl get pod -n kube-system — sort-by=.metadata.creationTimestamp | tac*

완료후에는 아래와 같이 정상적으로 업그레이드 된 것을 확인할 수 있습니다.

![kubectl get node -o wide](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*k6GY9k1Kz5LXAgh8AuXPAA.png) *kubectl get node -o wide*

## Automated Upgrades

클러스터 규모가 확장될 경우, Rancher의 `system-upgrade-controller`(SUC)를 도입하여 업그레이드 과정을 쿠버네티스 네이티브(Kubernetes-native) 방식으로 전환할 수 있습니다. 이 방식은 관리자가 호스트 환경에 개별 접속하는 대신, `Plan` 사용자 정의 리소스(CRD)를 사용하여 업그레이드 대상과 정책을 선언적으로 지정합니다.

`system-upgrade-controller`는 클러스터 내부에 배포되어 `Plan` 리소스를 모니터링합니다. 대상 버전(`version`)이 지정되거나, 지정된 릴리스 채널(`channel`)에서 새로운 버전이 감지되면, 컨트롤러는 노드 레이블을 기반으로 업그레이드를 실행할 일회성 작업 파드(Job Pod)를 생성합니다. 작업이 완료된 노드에는 완료 상태를 나타내는 레이블이 부여됩니다.

이를 위해 먼저 `system-upgrade-controller` 를 설치합니다.

```bash
kubectl apply -f https://github.com/rancher/system-upgrade-controller/releases/latest/download/crd.yaml -f https://github.com/rancher/system-upgrade-controller/releases/latest/download/system-upgrade-controller.yaml
```

![kubectl get deploy,pod,cm -n system-upgrade](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*bVmT9WpWjatsx8sYRMs54Q.png) *kubectl get deploy,pod,cm -n system-upgrade*

RKE2 업그레이드는 항상 서버 노드(Control Plane)가 에이전트 노드(Worker)보다 먼저 수행되어야 합니다. 이를 제어하기 위해 서버용 `Plan`과 에이전트용 `Plan`을 분리하여 작성합니다.

{% raw %}
```yaml
cat << EOF | kubectl apply -f -
apiVersion: upgrade.cattle.io/v1
kind: Plan
metadata:
  name: server-plan
  namespace: system-upgrade
spec:
  concurrency: 1
  cordon: true
  nodeSelector:
    matchExpressions:
    - key: node-role.kubernetes.io/control-plane
      operator: In
      values:
      - "true"
  serviceAccountName: system-upgrade
  upgrade:
    image: rancher/rke2-upgrade
  channel: https://update.rke2.io/v1-release/channels/latest  # version: v1.35.0+rke2r3 , curl -s https://update.rke2.io/v1-release/channels | jq .data
---
apiVersion: upgrade.cattle.io/v1
kind: Plan
metadata:
  name: agent-plan
  namespace: system-upgrade
spec:
  concurrency: 1
  cordon: true
  nodeSelector:
    matchExpressions:
    - key: node-role.kubernetes.io/control-plane
      operator: DoesNotExist
  prepare:
    args:
    - prepare
    - server-plan
    image: rancher/rke2-upgrade
  serviceAccountName: system-upgrade
  upgrade:
    image: rancher/rke2-upgrade
  channel: https://update.rke2.io/v1-release/channels/latest
EOF
```
{% endraw %}

에이전트 `Plan` 내부의 `prepare` 단계는 `server-plan`의 완료 상태를 확인하는 종속성 로직을 포함합니다. 이를 통해 클러스터 전체의 롤링 업그레이드 무결성을 보장합니다.

업그레이드 수행 시, system-upgrade 네임스페이스 생성된 Plan과 Job등을 확인할 수 있습니다.

![kubectl -n system-upgrade get plans -o wide](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*KxMHGvQf9YrSRT_2sMbagQ.png) *kubectl -n system-upgrade get plans -o wide*

![kubectl -n system-upgrade get jobs](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*94sWACVOTMMgmlogyqA1-w.png) *kubectl -n system-upgrade get jobs*

> 업그레이드를 수행하는 Job 파드는 호스트의 데몬을 교체해야 하므로 Host IPC/NET/PID 네임스페이스 사용, `CAP_SYS_BOOT` 권한, 그리고 호스트 루트 파일 시스템(`/`) 마운트 등 최고 수준의 권한이 필요합니다

이후 정상적으로 업그레이드 된 것을 확인할 수 있습니다.

![Upgrade completed](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ePo6PIYS5Vr8SJcfqTdVzg.png) *Upgrade completed*

## Cluster Disaster Recovery

쿠버네티스 컨트롤 플레인 컴포넌트(apiserver, etcd 등)는 API 스키마 변경 문제로 인해 **버전 다운그레이드가 지원되지 않습니다.** 업그레이드 과정 중 심각한 호환성 문제가 발생할 경우, 유일한 복구 방법은 업그레이드 직전에 백업된 etcd 스냅샷을 사용하여 클러스터를 이전 상태로 롤백하는 것입니다. 이를 위해 RKE2는 `rke2 etcd-snapshot` CLI를 내장하고 있으며, S3 호환 스토리지를 통한 외부 백업 기능을 제공합니다.
