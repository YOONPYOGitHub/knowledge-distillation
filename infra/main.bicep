// KD UI 배포: 2단계 (선행 리소스 → Container Apps)
//
// 사용:
//   1) 선행:   ./infra/deploy.sh prereqs      (Storage/ACR/Env + 권한)
//   2) 빌드:   ./scripts/build_and_push.sh <acrName> v1
//   3) 앱:     ./infra/deploy.sh apps         (Container Apps)
//
// 또는 한 번에:
//   ./infra/deploy.sh all
//
// chicken-and-egg 문제 (MI 가 ACR/Storage 권한 받기 전에 Container App 이 풀 시도)
// 를 피하기 위해 Container Apps 는 별도 모듈에서 권한 부여 이후 배포한다.

@description('리소스 이름 prefix (소문자/숫자, 3~12자)')
@minLength(3)
@maxLength(12)
param namePrefix string

@description('Azure 지역')
param location string = resourceGroup().location

@description('배포 모드: prereqs | apps | all')
@allowed(['prereqs', 'apps', 'all'])
param deployMode string = 'prereqs'

@description('Backend 이미지 (apps 또는 all 일 때 필요)')
param backendImage string = ''

@description('Frontend 이미지 (apps 또는 all 일 때 필요)')
param frontendImage string = ''

@description('Blob 컨테이너 이름')
param blobContainerName string = 'kd-results'

@description('Container Apps 최소 replica 수')
param minReplicas int = 0

@description('VNet 이 위치한 Resource Group')
param vnetResourceGroup string = 'rg-mgmt'

@description('사용할 VNet 이름')
param vnetName string = 'VNET-KRC'

@description('ACA infrastructure subnet (Microsoft.App/environments 위임 필요)')
param acaSubnetName string = 'snet-aca-infra'

@description('Private Endpoint 용 subnet')
param peSubnetName string = 'snet-pe'

@description('PE 검증 후 Storage public access 차단 (true) / 디버그 시 false')
param disableStoragePublic bool = true

var deployPrereqs = deployMode == 'prereqs' || deployMode == 'all'
var deployApps    = (deployMode == 'apps' || deployMode == 'all') && !empty(backendImage) && !empty(frontendImage)

// ---- 선행 리소스 (Storage / ACR / Env / 사전-생성 user-assigned MI) ----------

module prereqs 'modules/prereqs.bicep' = if (deployPrereqs) {
  name: 'prereqs'
  params: {
    namePrefix: namePrefix
    location: location
    blobContainerName: blobContainerName
    vnetResourceGroup: vnetResourceGroup
    vnetName: vnetName
    acaSubnetName: acaSubnetName
    peSubnetName: peSubnetName
    disableStoragePublic: disableStoragePublic
  }
}

// ---- Container Apps (이미지 적용 단계) ------------------------------------

module apps 'modules/apps.bicep' = if (deployApps) {
  name: 'apps'
  params: {
    namePrefix: namePrefix
    location: location
    backendImage: backendImage
    frontendImage: frontendImage
    blobContainerName: blobContainerName
    minReplicas: minReplicas
  }
}

output storageAccountName string = deployPrereqs ? prereqs.outputs.storageAccountName : ''
output acrLoginServer    string = deployPrereqs ? prereqs.outputs.acrLoginServer    : ''
output acrName           string = deployPrereqs ? prereqs.outputs.acrName           : ''
output frontendUrl       string = deployApps    ? apps.outputs.frontendUrl          : ''
output backendUrl        string = deployApps    ? apps.outputs.backendUrl           : ''
