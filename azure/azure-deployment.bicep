@description('environment id')
param envId string = 'TBD'


resource agent-database "Microsoft.App/containerApps@2023-03-01" = {
    name: it-support-chromadb
    location: 'northeurope'
    properties:
        environmentId: '${envId}'
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

resource agent-database 'Microsoft.Search/searchServices@2023-08-01' = {
  name: 'it-support-ai-search'        // must be globally unique within Azure
  location: 'northeurope'
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
    name: 'it-support-chatbot'
    location: westeurope //todo double check it
    properties: {
        environmentId: '${envId}'
        configuration: {
            ingress: {
                external: false
                targetPort: 8000
            }

            volumes: {
                name: 'chatbot-logs'
                azureFile: {
                    shareName: 'chatbot-logs'
                    storageAccountName: TBD //todo look if this is available in the current env
                }
            }
        }
        template: {
            containers: [
                {
                    name: 'it-support-chatbot'
                    image: TBD //look into it

                    env: [
                        {
                            name: 'CHROMADB_URL' / 'AISEARCH_URL'
                            value: TBD
                        }
                        {
                            name: 'OPENAI_KEY'
                            secretRef: 'openai-key'
                        }
                    ]

                    volumeMounts: [
                        {
                            name: 'chatbot-logs'
                            mountPath: '/app/logs'
                        }
                    ]

                    resources: {
                        cpu: 1 // might need 2 if we consider image porcessing
                        memory: '2Gi' // might need 4 Gi
                    }

                }
            ]

            scale: {
                minReplicas: 0 // adds serverless activity, so we save money. probably will take a while to boot up though
                maxReplicas: 2 // dk how much we want to scale it 
            }
        }
    }

}


