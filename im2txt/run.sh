# 模型文件夹或文件路径
CHECKPOINT_PATH="/home/widiot/workspace/tensorflow-space/tensorflow-gpu/practices/im2txt/model/newmodel.ckpt-2000000"
# 词汇文件
VOCAB_FILE="/home/widiot/workspace/tensorflow-space/tensorflow-gpu/practices/im2txt/data/word_counts.txt"
# 图片文件，多个图片用逗号隔开
IMAGE_FILE="/home/widiot/workspace/tensorflow-space/tensorflow-gpu/practices/im2txt/data/images/3.jpg"

# bazel编译
cd ..
bazel build -c opt //im2txt:run_inference

# 用参数调用编译后的文件
bazel-bin/im2txt/run_inference \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${IMAGE_FILE}