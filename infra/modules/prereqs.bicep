// 선행 리소스: Storage / ACR / Log Analytics / VNet 통합 Container Apps Environment
// + UAMI (Container Apps 가 ACR/Blob 접근에 사용)
// + Role assignments (UAMI → ACR/Storage)
// + Storage Private Endpoint + Private DNS Zone
//
// VNet 과 Private DNS zone 은 외부 RG 에 있는 것을 참조한다 (cross-RG).
// 더 이상 존재하지 않으면 배포는 실패하므로, 해당 리소스는 별도로 미리 준비해야 한다.

@minLength(3)
@maxLength(12)
param namePrefix string
param location string
param blobContainerName string

@description('VNet 이 위치한 Resource Group (외부 RG 참조)')
param vnetResourceGroup string

@description('사용할 VNet 이름')
param vnetName string

@description('ACA infrastructure subnet 이름 (Microsoft.App/environments 위임 필요, /23 권장)')
param acaSubnetName string

@description('Private Endpoint 용 subnet 이름')
param peSubnetName string

@description('Storage public network access 차단 여부 (PE 검증 후 true 권장)')
param disableStoragePublic bool = true

@description('Private DNS zone 이 위치한 Resource Group')
param dnsZoneResourceGroup string = vnetResourceGroup

var storageName = toLower('${namePrefix}st${uniqueString(resourceGroup().id)}')
var acrName     = toLower('${namePrefix}acr${uniqueString(resourceGroup().id)}')
var envName     = '${namePrefix}-env'
var logName     = '${namePrefix}-logs'
var uamiName    = '${namePrefix}-uami'
var peName      = '${namePrefix}-st-pe'
var dnsZoneName = 'privatelink.blob.${environment().suffixes.storage}'

var acaSubnetId = resourceId(vnetResourceGroup, 'Microsoft.Network/virtualNetworks/subnets', vnetName, acaSubnetName)
var peSubnetId  = resourceId(vnetResourceGroup, 'Microsoft.Network/virtualNetworks/subnets', vnetName, peSubnetName)
var vnetId      = resourceId(vnetResourceGroup, 'Microsoft.Network/virtualNetworks', vnetName)

// ---- UAMI (Container Apps 공용 ID) -----------------------------------------

resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: uamiName
  location: location
}

// ---- Storage ---------------------------------------------------------------

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    publicNetworkAccess: disableStoragePublic ? 'Disabled' : 'Enabled'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
    }
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: blobContainerName
  properties: { publicAccess: 'None' }
}

// ---- Private DNS Zone (cross-RG 참조; VNET-KRC 에 이미 link 됨) ----------

var dnsZoneId = resourceId(dnsZoneResourceGroup, 'Microsoft.Network/privateDnsZones', dnsZoneName)

// ---- Private Endpoint: Storage (blob) --------------------------------------

resource pe 'Microsoft.Network/privateEndpoints@2024-05-01' = {
  name: peName
  location: location
  properties: {
    subnet: { id: peSubnetId }
    privateLinkServiceConnections: [
      {
        name: '${peName}-conn'
        properties: {
          privateLinkServiceId: storage.id
          groupIds: [ 'blob' ]
        }
      }
    ]
  }
}

resource peDns 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = {
  parent: pe
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'blob'
        properties: { privateDnsZoneId: dnsZoneId }
      }
    ]
  }
}

// ---- ACR -------------------------------------------------------------------

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

// ---- Log Analytics + VNet 통합 Container Apps Environment ------------------

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
    vnetConfiguration: {
      infrastructureSubnetId: acaSubnetId
      internal: false
    }
  }
}

// ---- 권한 부여: UAMI → ACR (AcrPull) --------------------------------------

var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, uami.id, 'AcrPull')
  scope: acr
  properties: {
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
  }
}

// ---- 권한 부여: UAMI → Storage (Blob Data Contributor) --------------------

var blobContribRoleId = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'

resource blobContrib 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, uami.id, 'BlobContrib')
  scope: storage
  properties: {
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', blobContribRoleId)
  }
}

output storageAccountName string = storage.name
output acrLoginServer string = acr.properties.loginServer
output acrName string = acr.name
output uamiId string = uami.id
output uamiClientId string = uami.properties.clientId
output envId string = env.id
output privateEndpointId string = pe.id
