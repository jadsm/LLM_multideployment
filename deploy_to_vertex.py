import vertexai
from vertexai import model_garden

vertexai.init(project="imperial-410612", location="europe-west4")

model = model_garden.OpenModel("google/txgemma@txgemma-2b-predict")
endpoint = model.deploy(
  accept_eula=True,
  machine_type="ct5lp-hightpu-1t",
  serving_container_image_uri="us-docker.pkg.dev/vertex-ai-restricted/vertex-vision-model-garden-dockers/hex-llm-serve:20241210_2323_RC00",
  endpoint_display_name="google_txgemma-2b-predict-mg-one-click-deploy",
  model_display_name="google_txgemma-2b-predict-1780059895007",
  use_dedicated_endpoint=True,
  reservation_affinity_type="NO_RESERVATION",
)