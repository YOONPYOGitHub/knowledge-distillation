// Container Apps: backend (internal) + frontend (external)
// 선행 모듈에서 만든 UAMI / ACR / Storage / Env 를 참조한다.

@minLength(3)
@maxLength(12)
param namePrefix string
param location string
param backendImage string
param frontendImage string
param blobContainerName string
param minReplicas int

// prereqs 모듈에서 만든 리소스를 참조 (이름 규칙 일치)
var storageName = toLower('${namePrefix}st${uniqueString(resourceGroup().id)}')
var acrName     = toLower('${namePrefix}acr${uniqueString(resourceGroup().id)}')
var envName     = '${namePrefix}-env'
var uamiName    = '${namePrefix}-uami'

resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: uamiName
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageName
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: envName
}

// ---- Backend (internal ingress) -------------------------------------------

resource backend 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${namePrefix}-backend'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: false
        targetPort: 8000
        transport: 'auto'
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: uami.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'backend'
          image: backendImage
          resources: { cpu: json('2.0'), memory: '4Gi' }
          env: [
            { name: 'STORAGE_BACKEND', value: 'blob' }
            { name: 'AZURE_STORAGE_ACCOUNT', value: storage.name }
            { name: 'AZURE_BLOB_CONTAINER', value: blobContainerName }
            { name: 'AZURE_CLIENT_ID', value: uami.properties.clientId }
            { name: 'MODEL_CACHE_SIZE', value: '2' }
            { name: 'DEVICE', value: 'cpu' }
          ]
        }
      ]
      scale: { minReplicas: minReplicas, maxReplicas: 3 }
    }
  }
}

// ---- Frontend (external ingress) ------------------------------------------

resource frontend 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${namePrefix}-frontend'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8501
        transport: 'auto'
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: uami.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'frontend'
          image: frontendImage
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            {
              name: 'API_BASE_URL'
              value: 'https://${backend.properties.configuration.ingress.fqdn}'
            }
          ]
        }
      ]
      scale: { minReplicas: minReplicas, maxReplicas: 2 }
    }
  }
}

output backendUrl string = 'https://${backend.properties.configuration.ingress.fqdn}'
output frontendUrl string = 'https://${frontend.properties.configuration.ingress.fqdn}'
