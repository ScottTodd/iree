iree_fetch_artifact(
  NAME "model-matmul_1x256x2048_i8_i4_i32_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.30_1719457850/DOT_PRODUCT_JAX_1X256X2048XI8I4/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_1x256x2048_i8_i4_i32_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-matmul_256x256x2048_i8_i4_i32_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.30_1719457850/DOT_PRODUCT_JAX_256X256X2048XI8I4/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_256x256x2048_i8_i4_i32_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-matmul_1x256x2048_i8_i8_i32_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.30_1719457850/DOT_PRODUCT_JAX_1X256X2048XI8I8/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_1x256x2048_i8_i8_i32_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-matmul_256x256x2048_i8_i8_i32_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.30_1719457850/DOT_PRODUCT_JAX_256X256X2048XI8I8/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_256x256x2048_i8_i8_i32_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-PersonDetect_int8"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/person_detect.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_PersonDetect_int8.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-MobileNetV3Small_fp32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/MobileNetV3SmallStaticBatch.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_MobileNetV3Small_fp32.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-DeepLabV3_fp32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/deeplabv3.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_DeepLabV3_fp32.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-EfficientNet_int8"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/efficientnet_lite0_int8_2.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_EfficientNet_int8.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-MobileNetV1_fp32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v1_224_1.0_float.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_MobileNetV1_fp32.0_float.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-MobileNetV2_fp32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_MobileNetV2_fp32.0_224.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-MobileNetV2_int8"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224_quantized.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_MobileNetV2_int8.0_224_quantized.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-MobileSSD_fp32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/mobile_ssd_v2_float_coco.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_MobileSSD_fp32.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-PoseNet_fp32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/posenet.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_PoseNet_fp32.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-MobileBertSquad_fp16"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/mobilebertsquad.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_MobileBertSquad_fp16.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-MobileBertSquad_fp32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-float.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_MobileBertSquad_fp32.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-MobileBertSquad_int8"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-quant.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_MobileBertSquad_int8.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-EfficientNetV2STF"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/manual/EfficientNetV2STF_1af8c88f4e64e388a0c87bbeddcfb888084059df30cd631340d51794a0796e0f.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-MiniLML12H384Uncased"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/manual/MiniLML12H384Uncased_5aed9c3c3dfe8247ce76b74d518fa570b94dc0c3732631734d02ad70e4c74867.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-GPT2_117M_TF_1X4XI32"
  SOURCE_URL "https://storage.googleapis.com/iree-shared-files/tf_gpt2/static_input_seqlen5/stablehlo.mlir"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X4XI32.mlir"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-GPT2_117M_TF_1X1XI32"
  SOURCE_URL "https://storage.googleapis.com/iree-shared-files/tf_gpt2/static_input_seqlen1/stablehlo.mlir"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X1XI32.mlir"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertForMaskedLMTF"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/manual/BertForMaskedLMTF_e757a10b24f6ff83aaae0ceb5bb05d4efe9ff3e9931f8e9a29f12bc5c2e42b5e.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargeTF"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/manual/BertLargeTF_000793afb016fb3afc559304bcb3ba6cdb2df1825e8976ca236c07c12e4f65fa.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BERT_LARGE_JAX_384XI32_BATCH1"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.25_1709787220/BERT_LARGE_FP32_JAX_384XI32_BATCH1/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH1.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BERT_LARGE_JAX_384XI32_BATCH32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.25_1709787220/BERT_LARGE_FP32_JAX_384XI32_BATCH32/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH32.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BERT_LARGE_JAX_384XI32_BATCH64"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.25_1709787220/BERT_LARGE_FP32_JAX_384XI32_BATCH64/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH64.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-RESNET50_FP32_JAX_3X224X224XF32_BATCH1"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.25_1709787220/RESNET50_FP32_JAX_3X224X224XF32_BATCH1/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH1.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-RESNET50_FP32_JAX_3X224X224XF32_BATCH64"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.25_1709787220/RESNET50_FP32_JAX_3X224X224XF32_BATCH64/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH64.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-RESNET50_FP32_JAX_3X224X224XF32_BATCH128"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.25_1709787220/RESNET50_FP32_JAX_3X224X224XF32_BATCH128/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH128.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-T5_LARGE_FP32_JAX_512XI32_BATCH1"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.25_1709787220/T5_LARGE_FP32_JAX_512XI32_BATCH1/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH1.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-T5_LARGE_FP32_JAX_512XI32_BATCH16"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.25_1709787220/T5_LARGE_FP32_JAX_512XI32_BATCH16/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH16.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-T5_LARGE_FP32_JAX_512XI32_BATCH32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.25_1709787220/T5_LARGE_FP32_JAX_512XI32_BATCH32/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH32.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Vit_int8"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tflite/tflite_models_1698315913/VIT_CLASSIFICATION_INT8_TFLITE_3X224X224XINT8/model_int8.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Vit_int8.tflite"
  UNPACK
)
