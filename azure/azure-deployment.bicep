@secure()
param openaiKey string
@secure()
param jiraToken string
@secure()
param storageAccountName string

param location string = 'TBD'
param resourceGroupName string= 'TBD' // might not be needed
param envId string = 'subscription'

resource agentDatabase 'Microsoft.App/containerApps@2024-08-02' = {
    name: 'it-support-chromadb'
    location: location
    properties: {
        environmentId: envId
        configuration: {
            ingress: {
                external: false
                targetPort: 8000
            }
        }
        volumes: {
            name: 'chroma-data'
            azureFile: {
                shareName: 'chroma-data'
                storageAccountName: storageAccountName
            }
        }
        template: {
            containers: [
                {
                    name: 'chromadb'
                    image: 'repository?'
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
                    volumeMounts: [
                        {
                            name: 'chroma-data'
                            mountPath: '/chroma/chroma'
                        }
                    ]
                    resources: {
                        cpy: 1
                        memory: '2Gi'
                    }
                }
            ]
            scale: {
                minReplicas: 1
                maxReplicas: 1
            }
        }
    }
}

// ---- one of the options will be picked

// resource agent-database 'Microsoft.Search/searchServices@2023-08-01' = {
//   name: 'it-support-ai-search'        // must be globally unique within Azure
//   location: 'northeurope'
//   sku: {
//     name: 'basic'             // Options: Free, Basic, Standard, etc.
//     capacity: 1
//   }
//   properties: {
//     hostingMode: 'default'    // default or highDensity
//     replicaCount: 1           // for redundancy/scaling
//     partitionCount: 1         // number of partitions for scaling
//     publicNetworkAccess: 'Disabled'  // can set to 'Enabled' for public access
//     // optional: assign admin API key via keyVault later
//   }
// }

// ===

resource agentChatbot 'Microsoft.App/containerApps@2024-08-02' = {
    name: 'it-support-chatbot'
    location: location
    properties: {
        environmentId: envId
        configuration: {
            ingress: {
                external: false
                targetPort: 8000
            }
            secrets: [
                {
                    name: 'openai-key'
                    value: openaiKey
                }
                {
                    name: 'jira-token'
                    value: jiraToken
                }
            ]
            volumes: [
                {
                    name: 'chatbot-logs'
                    azureFile: {
                        shareName: 'chatbot-logs'
                        storageAccountName: storageAccountName
                    }
                }
            ]
        }
        template: {
            containers: [
                {
                    name: 'it-support-chatbot'
                    image: 'TBD' //look into it

                    env: [
                        {
                            name: 'OPENAI_KEY'
                            secretRef: 'openai-key'
                        }
                        {
                            name: 'JIRA_TOKEN'
                            valueRef: 'jira-token'
                        }
                        {
                            name: 'CHROMADB_URL'
                            value: 'http://it-support-chromadb.internal:8000'
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

resource agentFrontend 'Microsoft.App/containerApps@2024-08-02' = {
    name: 'it-support-frontend'
    location: location
    properties: {
        environmentId: envId
        configuration: {
            ingress: {
                external: true
                targetPort: 3000
            }
        }
        template: {
            containers: [
                {
                    name: 'it-support-frontend'
                    image: 'TBD'
                    env: [
                        {
                            name: 'NEXT_PUBLIC_API_URL'
                            value: 'http://it-support-chatbot.internal:8000'
                        }
                    ]
                    resources: {
                        cpu: 1
                        memory: '1Gi'
                    }
                }
            ]
            scale: {
                minReplicas: 0
                maxReplicas: 2
            }
        }
    }
}
