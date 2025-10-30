import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

// 序列长度：使用前N个字符预测下一个字符
const SEQUENCE_LENGTH = 3;

/**
 * 更新状态显示
 */
function updateStatus(message: string) {
  const statusEl = document.getElementById("status");
  if (statusEl) {
    statusEl.textContent = message;
  }
  console.log(message);
}

/**
 * 从 names.txt 读取名字数据
 */
async function loadNames(): Promise<string[]> {
  try {
    const response = await fetch("/names.txt");
    const text = await response.text();
    const allNames = text
      .split("\n")
      .map(name => name.trim().toLowerCase())
      .filter(name => name.length > 0 && name.length <= 10); // 只保留长度合理的名字

    console.log(`Found ${allNames.length} names in file`);

    // 🔥 使用所有数据，不进行切片
    console.log(`Using all ${allNames.length} names for training`);
    console.log("Sample names:", allNames.slice(0, 10));
    return allNames;
  } catch (error) {
    console.error("Error loading names:", error);
    return [];
  }
}

/**
 * 创建字符到索引和索引到字符的映射
 * 🔥 优化：直接使用固定的 26 个字母 + 结束标记，减少词汇表大小
 */
function createCharMappings() {
  // 固定使用 a-z 26 个字母 + 结束标记
  const chars = "abcdefghijklmnopqrstuvwxyz$".split("");
  const charToIndex = new Map<string, number>();
  const indexToChar = new Map<number, string>();

  chars.forEach((char, index) => {
    charToIndex.set(char, index);
    indexToChar.set(index, char);
  });

  const vocabSize = chars.length; // 27 个字符

  console.log(`Vocabulary size: ${vocabSize} (a-z + end marker)`);
  console.log("Characters:", chars.join(", "));

  return { charToIndex, indexToChar, vocabSize };
}

/**
 * 将名字转换为训练样本
 * 🔥 优化：不使用开始标记，只用结束标记，减少复杂度
 * 例如: "emma" -> 输入序列: ["emm", "mma", "ma$"], 输出: ["m", "a", "$"]
 */
function createTrainingData(names: string[], charToIndex: Map<string, number>) {
  const inputSequences: number[][] = [];
  const outputChars: number[] = [];

  names.forEach(name => {
    // 只添加结束标记（不用开始标记）
    const fullName = name + "$";

    // 创建滑动窗口序列
    for (let i = 0; i < fullName.length - SEQUENCE_LENGTH; i++) {
      const sequence = fullName.slice(i, i + SEQUENCE_LENGTH);
      const nextChar = fullName[i + SEQUENCE_LENGTH];

      // 将字符转换为索引
      const seqIndices = Array.from(sequence).map(c => charToIndex.get(c)!);
      const nextCharIndex = charToIndex.get(nextChar)!;

      inputSequences.push(seqIndices);
      outputChars.push(nextCharIndex);
    }
  });

  console.log(`Created ${inputSequences.length} training samples`);

  return { inputSequences, outputChars };
}

/**
 * 将训练数据转换为张量
 */
function convertToTensors(
  inputSequences: number[][],
  outputChars: number[],
  vocabSize: number
) {
  return tf.tidy(() => {
    // 输入: [样本数, 序列长度, 词汇表大小] (one-hot编码)
    const numSamples = inputSequences.length;

    // 创建one-hot编码的输入
    const inputTensor = tf.buffer([numSamples, SEQUENCE_LENGTH, vocabSize]);
    inputSequences.forEach((seq, i) => {
      seq.forEach((charIndex, j) => {
        inputTensor.set(1, i, j, charIndex);
      });
    });

    // 创建one-hot编码的输出
    const outputTensor = tf.buffer([numSamples, vocabSize]);
    outputChars.forEach((charIndex, i) => {
      outputTensor.set(1, i, charIndex);
    });

    return {
      inputs: inputTensor.toTensor(),
      labels: outputTensor.toTensor(),
    };
  });
}

/**
 * 创建LSTM模型用于生成名字
 * 🔥 优化：增强记忆性，添加正则化防止过拟合
 */
function createModel(vocabSize: number) {
  const model = tf.sequential();

  // 🔥 第一层 LSTM - 增强记忆能力，添加 L2 正则化
  model.add(
    tf.layers.lstm({
      units: 128,
      inputShape: [SEQUENCE_LENGTH, vocabSize],
      returnSequences: true,
      recurrentDropout: 0.1, // 🔥 LSTM 内部 dropout
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }), // 🔥 L2 正则化
    })
  );

  // 🔥 第二层 LSTM
  model.add(
    tf.layers.lstm({
      units: 64,
      returnSequences: false,
      recurrentDropout: 0.1,
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    })
  );

  // Dense 层
  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    })
  );

  // Dropout 防止过拟合
  model.add(tf.layers.dropout({ rate: 0.3 })); // 🔥 增加 dropout 比率

  // 输出层
  model.add(
    tf.layers.dense({
      units: vocabSize,
      activation: "softmax",
    })
  );

  return model;
}

/**
 * 训练模型
 */
async function trainModel(
  model: tf.Sequential,
  inputs: tf.Tensor,
  labels: tf.Tensor
) {
  model.compile({
    optimizer: tf.train.adam(0.002), // 🔥 降低学习率，防止过拟合
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const batchSize = 256; // 🔥 增大 batch size
  const epochs = 6; // 🔥 减少到 6 epochs，避免过拟合

  console.log("Starting training...");

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    validationSplit: 0.15, // 🔥 增加验证集比例，更好监控过拟合
    yieldEvery: "batch",
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "acc", "val_loss", "val_acc"],
      { height: 300, callbacks: ["onEpochEnd"] }
    ),
  });
}

/**
 * 生成新的名字
 * 🔥 优化：增加长度控制，避免生成过长名字
 */
function generateName(
  model: tf.Sequential,
  charToIndex: Map<string, number>,
  indexToChar: Map<number, string>,
  vocabSize: number,
  maxLength: number = 10, // 🔥 减小默认最大长度
  temperature: number = 1.0
): string {
  // 从随机的常见起始字母开始（英文名常见首字母）
  const startLetters = "abcdefghjklmnprstw";
  const firstLetter =
    startLetters[Math.floor(Math.random() * startLetters.length)];
  let name = firstLetter;

  // 🔥 添加结束标记的索引
  const endMarkerIndex = charToIndex.get("$")!;

  while (name.length < maxLength) {
    // 获取最后SEQUENCE_LENGTH个字符
    let sequence = name.slice(-SEQUENCE_LENGTH);
    while (sequence.length < SEQUENCE_LENGTH) {
      sequence = name[0] + sequence; // 用第一个字母填充
    }

    // 转换为one-hot编码
    const inputTensor = tf.tidy(() => {
      const buffer = tf.buffer([1, SEQUENCE_LENGTH, vocabSize]);
      Array.from(sequence).forEach((char, i) => {
        const index = charToIndex.get(char);
        if (index !== undefined) {
          buffer.set(1, 0, i, index);
        }
      });
      return buffer.toTensor();
    });

    // 预测下一个字符
    const prediction = model.predict(inputTensor) as tf.Tensor;

    // 应用温度参数并采样
    const nextCharIndex = tf.tidy(() => {
      const probsArray = Array.from(prediction.dataSync());

      // 🔥 长度惩罚：名字越长，越倾向于结束
      if (name.length >= 6) {
        // 增加结束标记的概率
        const boost = Math.min((name.length - 5) * 0.15, 0.6);
        probsArray[endMarkerIndex] += boost;

        // 重新归一化
        const sum = probsArray.reduce((a: number, b: number) => a + b, 0);
        for (let i = 0; i < probsArray.length; i++) {
          probsArray[i] /= sum;
        }
      }

      // 温度调整
      const scaledProbs = probsArray.map((p: number) =>
        Math.pow(p, 1 / temperature)
      );
      const scaledSum = scaledProbs.reduce((a: number, b: number) => a + b, 0);
      const normalizedProbs = scaledProbs.map((p: number) => p / scaledSum);

      // 采样
      const rand = Math.random();
      let cumulative = 0;
      for (let i = 0; i < normalizedProbs.length; i++) {
        cumulative += normalizedProbs[i];
        if (rand < cumulative) {
          return i;
        }
      }
      return normalizedProbs.length - 1;
    });

    inputTensor.dispose();
    prediction.dispose();

    const nextChar = indexToChar.get(nextCharIndex)!;

    // 如果遇到结束标记，停止生成
    if (nextChar === "$") {
      break;
    }

    name += nextChar;
  }

  // 🔥 如果名字太短（少于3个字符），重新生成
  if (name.length < 3) {
    return generateName(
      model,
      charToIndex,
      indexToChar,
      vocabSize,
      maxLength,
      temperature
    );
  }

  return name;
}

/**
 * 生成多个名字并显示
 */
function generateNames(
  model: tf.Sequential,
  charToIndex: Map<string, number>,
  indexToChar: Map<number, string>,
  vocabSize: number,
  count: number = 20
) {
  const names: string[] = [];

  console.log("\n🎉 Generated Names:");
  console.log("=".repeat(50));

  for (let i = 0; i < count; i++) {
    // 🔥 使用更保守的温度值，提高质量
    const temperature = 0.6 + Math.random() * 0.5; // 0.6 到 1.1
    const name = generateName(
      model,
      charToIndex,
      indexToChar,
      vocabSize,
      10, // 🔥 限制最大长度为 10
      temperature
    );

    if (name.length > 0) {
      // 首字母大写
      const capitalizedName = name.charAt(0).toUpperCase() + name.slice(1);
      names.push(capitalizedName);
      console.log(
        `${i + 1}. ${capitalizedName} (temp: ${temperature.toFixed(2)})`
      );
    }
  }

  console.log("=".repeat(50));

  // 在页面上显示生成的名字
  displayGeneratedNames(names);

  return names;
}

/**
 * 在页面上显示生成的名字
 */
function displayGeneratedNames(names: string[]) {
  const container = document.createElement("div");
  container.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    max-width: 300px;
    max-height: 80vh;
    overflow-y: auto;
    z-index: 1000;
  `;

  const title = document.createElement("h3");
  title.textContent = "🎉 Generated Names";
  title.style.marginTop = "0";
  container.appendChild(title);

  const list = document.createElement("ul");
  list.style.cssText = "padding-left: 20px; line-height: 1.6;";

  names.forEach(name => {
    const item = document.createElement("li");
    item.textContent = name;
    list.appendChild(item);
  });

  container.appendChild(list);

  // 移除旧的容器
  const oldContainer = document.querySelector("#generated-names-container");
  if (oldContainer) {
    oldContainer.remove();
  }

  container.id = "generated-names-container";
  document.body.appendChild(container);
}

/**
 * 主函数
 */
async function run() {
  console.log("🚀 Starting Name Generator Training...");
  updateStatus("🚀 Starting Name Generator Training...");

  // 1. 加载名字数据
  updateStatus("📖 Loading names from names.txt...");
  const names = await loadNames();

  if (names.length === 0) {
    console.error("No names loaded!");
    updateStatus("❌ Error: No names loaded!");
    return;
  }

  updateStatus(`✅ Loaded ${names.length} names`);

  // 2. 创建字符映射
  updateStatus("🔤 Creating character mappings...");
  const { charToIndex, indexToChar, vocabSize } = createCharMappings();
  updateStatus(`✅ Vocabulary size: ${vocabSize} characters (a-z + end)`);

  // 3. 创建训练数据
  updateStatus("🔨 Creating training sequences...");
  const { inputSequences, outputChars } = createTrainingData(
    names,
    charToIndex
  );
  updateStatus(`✅ Created ${inputSequences.length} training samples`);

  // 4. 转换为张量
  updateStatus("🧮 Converting data to tensors... (this may take a moment)");

  // 🔥 添加性能监控
  console.time("Tensor conversion");
  const { inputs, labels } = convertToTensors(
    inputSequences,
    outputChars,
    vocabSize
  );
  console.timeEnd("Tensor conversion");

  console.log("Input shape:", inputs.shape);
  console.log("Output shape:", labels.shape);
  console.log("Memory:", tf.memory());
  updateStatus(
    `✅ Tensor shapes: Input ${inputs.shape}, Output ${labels.shape}`
  );

  // 5. 创建模型
  updateStatus("🏗️ Building LSTM neural network...");
  const model = createModel(vocabSize);
  tfvis.show.modelSummary({ name: "Model Summary" }, model);
  updateStatus("✅ Model architecture created");

  // 6. 训练模型
  updateStatus("🎓 Training model... This may take a few minutes.");
  await trainModel(model, inputs, labels);
  console.log("✅ Training completed!");
  updateStatus("✅ Training completed successfully!");

  // 7. 生成新名字
  updateStatus("🎨 Generating creative new names...");
  generateNames(model, charToIndex, indexToChar, vocabSize, 20);
  updateStatus("🎊 Done! Check out the generated names →");

  // 清理张量
  inputs.dispose();
  labels.dispose();

  console.log("🎊 All done! Check the generated names on the right side.");
}

// 页面加载完成后运行
document.addEventListener("DOMContentLoaded", run);
