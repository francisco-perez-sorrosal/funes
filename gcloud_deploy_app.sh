export GCP_PROJECT=$(gcloud config get-value project)
export GCP_REGION='us-west1'

export AR_REPO='fperez-gcloud-stupid-sailor-twift-docker-repo'
export SERVICE_NAME='stupid-sailor-twift-streamlit-app' # This is the name of our Application and Cloud Run service. Change it if you'd like.
export CUSTOMER_MANAGED_KEY="projects/${GCP_PROJECT}/locations/${GCP_REGION}/keyRings/training-key/cryptoKeys/vertexai-notebooks-key-${GCP_REGION}"
echo "Current CKEM key is ${CUSTOMER_MANAGED_KEY}"

gcloud run deploy "$SERVICE_NAME" \
  --port=8080 \
  --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
  --allow-unauthenticated \
  --region=$GCP_REGION \
  --platform=managed  \
  --project=$GCP_PROJECT \
  --clear-key \
  --network="vpc-cresearch-gcp" --subnet="projects/${GCP_PROJECT}/regions/${GCP_REGION}/subnetworks/subnet-1" \
  --set-env-vars=GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION
