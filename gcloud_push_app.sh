export GCP_PROJECT=$(gcloud config get-value project)
export GCP_REGION='us-west1'

export AR_REPO='fperez-gcloud-stupid-sailor-twift-docker-repo'
export SERVICE_NAME='stupid-sailor-twift-streamlit-app' # This is the name of our Application and Cloud Run service. Change it if you'd like.

#make sure you are in the active directory for 'gemini-streamlit-cloudrun'
gcloud artifacts repositories create "$AR_REPO" --location="$GCP_REGION" --repository-format=Docker
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev"
gcloud builds submit --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"
