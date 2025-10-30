# 🎭 AI 英文名字生成器

基于 TensorFlow.js 的深度学习模型，能够学习并生成全新的英文名字。

## 📋 项目说明

这个项目使用 LSTM (长短期记忆网络) 从 `names.txt` 文件中学习超过 32,000 个英文名字的模式，然后生成创意性的新名字。

### 🏗️ 模型架构

```
输入层 (字符序列，one-hot 编码)
    ↓
LSTM 层 (128 units) - 学习序列模式
    ↓
Dense 层 (64 units, ReLU) - 特征提取
    ↓
Dropout (0.2) - 防止过拟合
    ↓
Dense 层 (32 units, ReLU) - 进一步抽象
    ↓
输出层 (vocab_size, Softmax) - 预测下一个字符
```

### 🔑 关键特性

- **序列学习**: 使用前 3 个字符预测下一个字符
- **LSTM 网络**: 能够记住长期依赖关系
- **温度采样**: 控制生成名字的创造性和多样性
- **可视化训练**: 实时查看损失和准确率曲线

## 🚀 运行项目

### 方式 1: 运行名字生成器

在 `package.json` 中修改 dev 脚本：

```json
"scripts": {
  "dev": "vite --open index2.html"
}
```

然后运行：

```bash
pnpm dev
```

### 方式 2: 运行汽车 MPG 预测（原项目）

```bash
# 保持默认配置
pnpm dev
```

## 📁 文件说明

- `src/main2.ts` - 名字生成器主代码
- `src/names.txt` - 训练数据（32,000+ 英文名字）
- `index2.html` - 名字生成器页面
- `src/main.ts` - 汽车 MPG 预测模型
- `index.html` - MPG 预测页面

## 🎯 代码亮点解析

### 1. 数据预处理

```typescript
// 为每个名字添加开始(^)和结束($)标记
const fullName = "^" + name + "$";

// 创建滑动窗口序列
// 例如: "^emma$" → ["^em","emm","mma","ma$"]
for (let i = 0; i < fullName.length - SEQUENCE_LENGTH; i++) {
  const sequence = fullName.slice(i, i + SEQUENCE_LENGTH);
  const nextChar = fullName[i + SEQUENCE_LENGTH];
}
```

### 2. One-Hot 编码

将字符转换为向量表示，例如：

- 'a' → [1, 0, 0, ..., 0]
- 'b' → [0, 1, 0, ..., 0]

### 3. LSTM 层的作用

```typescript
model.add(
  tf.layers.lstm({
    units: 128,
    inputShape: [SEQUENCE_LENGTH, vocabSize],
    returnSequences: false, // 只返回最后的输出
  })
);
```

LSTM 能够：

- 记住字符序列的长期模式
- 学习名字的结构规律（如常见的字母组合）
- 捕捉位置相关的特征（如首字母通常是大写辅音）

### 4. 温度采样

```typescript
const temperature = 0.5 + Math.random() * 0.8; // 0.5 到 1.3

// temperature 越低 → 越保守、越像训练数据
// temperature 越高 → 越创新、越随机
const scaledPrediction = prediction.div(temperature);
```

### 5. 名字生成过程

```
开始: "^"
预测: "^" → "e" → "^e"
预测: "^e" → "m" → "^em"
预测: "em" → "m" → "^emm"
预测: "mm" → "a" → "^emma"
预测: "ma" → "$" → 停止
结果: "emma"
```

## 📊 与 main.ts 的对比

| 特性       | main.ts (MPG 预测) | main2.ts (名字生成)      |
| ---------- | ------------------ | ------------------------ |
| 任务类型   | **回归问题**       | **序列生成问题**         |
| 网络类型   | Dense (全连接)     | LSTM (循环神经网络)      |
| 激活函数   | ReLU + 无          | ReLU + Softmax           |
| 损失函数   | MSE (均方误差)     | Categorical Crossentropy |
| 输入格式   | 单个数值           | 字符序列 (one-hot)       |
| 输出格式   | 单个数值           | 概率分布                 |
| 数据归一化 | Min-Max Scaling    | One-Hot Encoding         |

## 🎨 生成结果示例

训练完成后，模型可能生成类似这样的名字：

- Aria
- Emilia
- Lila
- Maren
- Nora
- Oliviana
- Sophia
- Zara

## 🔧 调优建议

### 提高生成质量：

1. **增加训练轮数**

```typescript
const epochs = 100; // 从 50 增加到 100
```

2. **调整 LSTM 单元数**

```typescript
units: 256, // 从 128 增加到 256
```

3. **添加更多层**

```typescript
model.add(tf.layers.lstm({ units: 128, returnSequences: true }));
model.add(tf.layers.lstm({ units: 64, returnSequences: false }));
```

4. **调整温度参数**

```typescript
// 更保守的生成
const temperature = 0.3;

// 更创新的生成
const temperature = 1.5;
```

## 📚 学习资源

- [TensorFlow.js 官方文档](https://www.tensorflow.org/js)
- [LSTM 原理](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [字符级语言模型](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## 🎓 技术要点

1. **为什么用 LSTM 而不是 Dense？**

   - Dense 层没有记忆，无法处理序列
   - LSTM 有记忆单元，能记住之前的字符

2. **为什么用 Softmax？**

   - 这是多分类问题（预测下一个字符是 a-z 中的哪一个）
   - Softmax 输出每个字符的概率

3. **为什么用 Categorical Crossentropy？**
   - 适合多分类问题
   - 能有效衡量预测概率分布与真实分布的差异

## 🚀 扩展想法

- 添加"生成更多"按钮
- 支持自定义序列长度
- 支持从用户输入的前缀生成
- 添加名字评分系统
- 训练不同风格的名字（日本名、中文名等）

---

**作者**: AI Assistant
**技术栈**: TypeScript + TensorFlow.js + Vite
**灵感来源**: Andrej Karpathy 的字符级 RNN
