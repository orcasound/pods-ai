# LiveInferenceSystem Container

`LiveInferenceSystem/` packages `src/LiveInferenceOrchestrator.py` as a Docker container for
production deployment to Azure Kubernetes Service (AKS), following the same pattern used by
[OrcaHello's InferenceSystem](https://github.com/orcasound/orcahello/tree/main/InferenceSystem).
The two containers can run side-by-side in the same Kubernetes cluster without conflicts.

## Quick Start

Build the image from the repo root (requires the `external/orcahello` submodule):

```bash
git submodule update --init external/orcahello
docker build -f LiveInferenceSystem/Dockerfile -t pods-ai-live-inference-system .
```

> **macOS M-series:** prefix with `docker buildx build --platform linux/amd64`

Run locally by mounting an orchestrator config at `/config/config.yml`:

```bash
# Linux/Mac
docker run --rm -it --env-file .env \
  -v $PWD/LiveInferenceSystem/tests/orch_configs/LiveHLS/LiveHLS_OrcasoundLab.yml:/config/config.yml \
  pods-ai-live-inference-system \
  --max_live_iterations 2

# Windows
docker run --rm -it --env-file .env ^
  -v %cd%/LiveInferenceSystem/tests/orch_configs/LiveHLS/LiveHLS_OrcasoundLab.yml:/config/config.yml ^
  pods-ai-live-inference-system ^
  --max_live_iterations 2
```

The `.env` file should contain Azure credentials (see `LiveInferenceOrchestrator.py` for required
environment variables).

## Deployment

In production each hydrophone location runs as a separate deployment in its own Kubernetes
namespace.  The `LiveInferenceSystem/deploy/` directory contains the Kubernetes manifests:

- `<location>.yaml` — deployment spec
- `<location>-configmap.yaml` — hydrophone-specific orchestrator configuration

To release a new container image, push a tag of the form `LiveInferenceSystem.v#.#.#`.
This triggers the `LiveInferenceSystem-deploy` workflow, which builds the image and pushes it to
`orcaconservancycr.azurecr.io/pods-ai-live-inference-system`.

To deploy to a hydrophone location:

```bash
NAMESPACE=orcasound-lab  # or andrews-bay, bush-point, etc.
kubectl apply -f LiveInferenceSystem/deploy/$NAMESPACE-configmap.yaml
# Scale to 0 first — required by the Recreate strategy on memory-constrained nodes
# so that the old pod is fully terminated before the new pod starts.
kubectl scale deployment pods-ai-inference-system -n $NAMESPACE --replicas=0
kubectl apply -f LiveInferenceSystem/deploy/$NAMESPACE.yaml
```

To add a new hydrophone location, create `deploy/<namespace>-configmap.yaml` and
`deploy/<namespace>.yaml` using an existing pair as a template, then create the namespace and
secret:

```bash
kubectl create namespace <namespace>
kubectl create secret generic pods-ai-inference-system -n <namespace> \
    --from-literal=AZURE_COSMOSDB_PRIMARY_KEY='<key>' \
    --from-literal=AZURE_STORAGE_CONNECTION_STRING='<string>' \
    --from-literal=INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING='<string>'
```

## Architecture

The timestamp correction implementation follows the architecture described in the [aifororcas-livesystem](https://github.com/orcasound/aifororcas-livesystem):

- Uses `DateRangeHLSStream` approach to download audio from specific time ranges
- Downloads from Orcasound S3 buckets: `s3-us-west-2.amazonaws.com/audio-orcasound-net/`
- Processes HLS streams with m3u8 playlists
- Uses FFmpeg for audio format conversion
- Returns `local_confidences` array with scores for each segment

## Example Configuration

Similar to [aifororcas-livesystem config files](https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/config/Test/Positive/FastAI_DateRangeHLS_AndrewsBay.yml):

```yaml
model_type: "FastAI"
model_local_threshold: 0.5
model_global_threshold: 3
model_path: "./model"
model_name: "model.pkl"
```

## GitHub CI configuration

The following repository secrets must be configured using information obtained
from HuggingFace:

* HF_TOKEN — Get this from https://huggingface.co/settings/tokens after logging in as the account used to publish the model (e.g., "davethaler").  This is used by train_model.yml.

or from portal.azure.com:

* COSMOS_KEY — "aifororcasmetadatastore" CosmosDB account → "Keys" → "Read-only Keys" → primary key.  This is used by bootstrap make_csv.py and train_model.yml.
* AZURE_COSMOSDB_PRIMARY_KEY — "aifororcasmetadatastore" CosmosDB account → "Keys" → "Read-write Keys" → primary key.  This is used by LiveInferenceOrchestrator.py.
* AZURE_STORAGE_CONNECTION_STRING — "livemlaudiospecstorage" storage account. See the "Connection String" section in [these instructions](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?tabs=connection-string%2Croles-azure-portal%2Csign-in-azure-cli&pivots=blob-storage-quickstart-scratch#authenticate-to-azure-and-authorize-access-to-blob-data).  This is used by LiveInferenceOrchestrator.py.
* INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING — "InferenceSystemInsights" Application Insights → "Overview" → connection string.  This is used by LiveInferenceOrchestrator.py.
* ACR_USERNAME — "orcaconservancycr" Container registry → "Access keys" → "Username".  This is used by LiveInferenceSystem-deploy.yaml.
* ACR_PASSWORD — "orcaconservancycr" Container registry → "Access keys" → "password".  This is used by LiveInferenceSystem-deploy.yaml.
* ACR_REGISTRY — "orcaconservancycr" Container registry → "Access keys" → "Registry name".  This is used by LiveInferenceSystem-deploy.yaml.
* KUBE_CONFIG — This is used by LiveInferenceSystem-deploy-configmaps.yaml.  To obtain the KUBE_CONFIG value, run the following:

```
az aks get-credentials --resource-group LiveSRKWNotificationSystem --name inference-system-AKS --admin --file kubeconfig
```

This produces a file named `kubeconfig`, the contents of which can be used as the KUBE_CONFIG value.
