import { pipeline } from "@xenova/transformers";

// 1. 初始化模型
const extractor = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2');

// 輔助函式：計算兩個向量的餘弦相似度
// 公式：(A · B) / (||A|| * ||B||)
// 如果向量已經過正規化 (Normalize)，相似度就是純粹的內積 (Dot Product)
function cosineSimilarity(vecA: number[], vecB: number[]): number {
    let dotProduct = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i]! * vecB[i]!;
    }
    return dotProduct; // 因為模型輸出的向量已 normalize，這裡直接回傳內積即可
}

// 2. 準備假資料 (模擬資料庫中的資料)
const mockData = [
    { word: "開心", videoUrl: "url_happy" },
    { word: "難過", videoUrl: "url_sad" },
    { word: "憂鬱", videoUrl: "url_sad" },
    { word: "天氣很好", videoUrl: "url_weather" },
    { word: "肚子餓了", videoUrl: "url_hungry" },
    { word: "你好嗎", videoUrl: "url_hello" },
    { word: "悲慘", videoUrl: "url_silence" },
];

console.log("正在初始化資料向量...");
const dataWithEmbeddings = await Promise.all(mockData.map(async (item) => {
    const output = await extractor(item.word, { pooling: 'mean', normalize: true });
    return {
        ...item,
        embedding: Array.from(output.data) as number[]
    };
}));

// 3. 搜尋函式
async function localSemanticSearch(query: string) {
    // 將搜尋詞轉為向量
    const output = await extractor(query, { pooling: 'mean', normalize: true });
    const queryVector = Array.from(output.data) as number[];

    // 計算每一筆資料與搜尋詞的相似度
    const results = dataWithEmbeddings.map(item => {
        return {
            word: item.word,
            videoUrl: item.videoUrl,
            similarity: cosineSimilarity(queryVector, item.embedding)
        };
    });

    // 依照相似度排序 (從高到低)
    return results.sort((a, b) => b.similarity - a.similarity).slice(0, 3);
}
const queryText = "好爽";
console.log(`\n搜尋關鍵字: "${queryText}"`);

const searchResults = await localSemanticSearch(queryText);
console.table(searchResults);
