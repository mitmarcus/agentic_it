resource agent-database "Microsoft.App/containerApps@2023-03-01" = {
    name: it-support-chromadb
    location: 'northeurope'
    properties:
        environmentId: TBD
        configuration: {
            ingress: {
                external: false
                targetPort: 8000
            }
        }
        volumes:{
            name: 'chroma-data'
            azureFile: {
                shareName: 'chroma-data'
                storageAccountName: TBD //todo look if this is available in the current env
            }
        }
        template: {
            containers: [
                {
                    name: 'chromadb'
                    image: 'ghcr.io/chroma-core/chroma:latest'
                    env: [
                        {
                            name: 'IS_PERSISTENT'
                            secretRef: 'TRUE'
                        }
                        {
                            name: 'ANONYMIZED_TELEMETRY'
                            value: 'TRUE'
                        }
                    ]
                }
                volumeMounts: [
                    name: 'chroma-data'
                    mountPath: '/chroma/chroma'
                ]
            ]
            scale: {
                minReplicas: 1
                maxReplicas: 1
            }
        }
}

// ---- one of the options will be picked

resource aiSearch 'Microsoft.Search/searchServices@2023-08-01' = {
  name: 'my-ai-search'        // must be globally unique within Azure
  location: 'westeurope'
  sku: {
    name: 'basic'             // Options: Free, Basic, Standard, etc.
    capacity: 1
  }
  properties: {
    hostingMode: 'default'    // default or highDensity
    replicaCount: 1           // for redundancy/scaling
    partitionCount: 1         // number of partitions for scaling
    publicNetworkAccess: 'Disabled'  // can set to 'Enabled' for public access
    // optional: assign admin API key via keyVault later
  }
}

// ===

resource agent-chatbot "Microsoft.App/containerApps@2023-03-01" = {
    dependsOn: [agent-database]
    name: 'chatbot'
}

resource agent-frontend "Microsoft.App/containerApps@2023-03-01" = {
    dependsOn: [agent-chatbot]
    name: 'frontend'
}
