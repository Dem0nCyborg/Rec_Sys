import React, { useEffect, useState } from 'react';

function App() {
  const [userId, setUserId] = useState();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  // A robust feature: Give the professor quick-test users
  const sampleUsers = ['A10JB7YPWZGRF4', 'A10ZJZNO4DAVB'];
  const [selectedImages, setSelectedImages] = useState({});
  const [w1, setW1] = useState(1.0); // accuracy
  const [w2, setW2] = useState(0.0); // diversity
  const [w3, setW3] = useState(0.0); // novelty
  const [alpha, setAlpha] = useState(0.0); // sentiment
  const [selectedItem, setSelectedItem] = useState(null);
  

  const getPerformance = () => {
    if (!data?.model_performance) return null;

    if (alpha === 0) {
      return data.model_performance.baseline;
    }

    return data.model_performance.alpha_sweep.find(
      (item) => item.alpha === alpha
    );
  };

  const [pref, setPref] = useState(getPerformance());

  const fetchRecs = async (idToFetch) => {
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/recommend/${idToFetch}?w1=${w1}&w2=${w2}&w3=${w3}&alpha=${alpha}`);
      if (!res.ok) throw new Error("Backend not responding");
      const json = await res.json();
      setData(json);
      setPref(getPerformance());
    } catch (err) {
      alert("Error: Make sure your FastAPI backend is running!");
    } finally {
      setLoading(false);
    }
  };
  // useEffect(() => {
  //   if (userId) fetchRecs(userId);
  // }, [w1, w2, w3]); // Refetch when weights change

  return (
    // Apple-style light background (#f5f5f7)
    <div className="min-h-screen bg-[#f5f5f7] text-slate-800 p-6 md:p-12 font-sans selection:bg-blue-200 overflow-hidden relative">
      
      {/* Soft Pastel Background Blobs for that Apple Vibrancy effect */}
      <div className="absolute top-[-5%] left-[-5%] w-96 h-96 bg-blue-300/40 rounded-full blur-[100px] pointer-events-none"></div>
      <div className="absolute bottom-[-10%] right-[-5%] w-96 h-96 bg-pink-300/30 rounded-full blur-[120px] pointer-events-none"></div>
      <div className="absolute top-[40%] left-[30%] w-80 h-80 bg-teal-200/40 rounded-full blur-[100px] pointer-events-none"></div>

      <div className="max-w-5xl mx-auto relative z-10">
        
        {/* Header & Search */}
        <header className="grid md:grid-cols-2 gap-8 items-center mb-8">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold">
              EchoRec <span className="text-blue-500">AI</span>
            </h1>

            <p className="text-slate-500 font-medium text-sm mt-3 uppercase tracking-[0.15em]">
              Neural Recommendation Engine
            </p>

            <p className="mt-2 text-slate-500 text-sm max-w-sm">
              Discover personalized products using deep learning, controllable ranking,
              and explainable AI.
            </p>
          </div>
          
          <div className="flex flex-col gap-3">

            {/* Top row */}
            <div className="flex items-center gap-3">

              {/* Search */}
              <div className="flex bg-white/60 p-1.5 rounded-xl border backdrop-blur-xl shadow-sm">
                <input 
                  type="text" 
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  className="bg-transparent px-3 py-1.5 w-40 text-sm focus:outline-none"
                  placeholder="User ID..."
                />
                <button 
                  onClick={() => fetchRecs(userId)}
                  className="bg-blue-500 text-white px-4 py-1.5 rounded-lg text-sm"
                >
                  Go
                </button>
                 {/* Quick Select */}
                  <div className="flex gap-2 text-xs font-medium text-slate-400">
                    <span>  </span>
                    {sampleUsers.map(user => (
                      <button 
                        key={user} 
                        onClick={() => { setUserId(user); fetchRecs(user); }}
                        className="hover:text-blue-500 transition-colors cursor-pointer"
                      >
                        {user}
                      </button>
                    ))}
                  </div>
              </div>

            </div>

            {/* Controls INLINE (horizontal) */}
            <div className="grid grid-cols-2 gap-4 bg-white/60 px-4 py-4 rounded-2xl border backdrop-blur-xl shadow-sm">

            {/* Accuracy */}
            <div className="flex flex-col text-xs">
              <div className="flex justify-between">
                <span>Accuracy</span>
                <span>{w1.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={w1}
                onChange={(e) => setW1(parseFloat(e.target.value))}
              />
            </div>

            {/* Diversity */}
            <div className="flex flex-col text-xs">
              <div className="flex justify-between">
                <span>Diversity</span>
                <span>{w2.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={w2}
                onChange={(e) => setW2(parseFloat(e.target.value))}
              />
            </div>

            {/* Novelty */}
            <div className="flex flex-col text-xs">
              <div className="flex justify-between">
                <span>Novelty</span>
                <span>{w3.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={w3}
                onChange={(e) => setW3(parseFloat(e.target.value))}
              />
            </div>

            {/* Sentiment */}
            <div className="flex flex-col text-xs">
              <div className="flex justify-between">
                <span>Sentiment</span>
                <span>{alpha.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="0.3"
                step="0.1"
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
              />
            </div>

          </div>
             

        </div>
        </header>
            {/* Recommendations List - Rendering AI Reasoning for ALL items */}
            <p className="text-sm text-slate-500">
                Showing top 10 recommendations
              </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-4">
              
              {data?.recommendations.map((item, i) => {

                const images = item.imageURLHighRes?.length
                  ? item.imageURLHighRes
                  : ["https://t3.ftcdn.net/jpg/05/04/28/96/360_F_504289605_zehJiK0tCuZLP2MdfFBpcJdOVxKLnXg1.jpg"];

                const selectedImage = selectedImages[i] || images[0];
                
                // const images = item.imageURLHighRes?.length
                //   ? item.imageURLHighRes
                //   : ["https://via.placeholder.com/150"];

                // Better label system
                const label =
                  item.score > 5 ? "🔥 Top Pick" :
                  item.score > 4.7 ? "✨ Great Match" :
                  "👍 Good Match";

                return (
                  <div
                    key={i}
                    onClick={() => setSelectedItem(item)}
                    className="bg-white/70 border border-white p-4 rounded-2xl 
                    hover:shadow-xl hover:scale-[1.02] transition-all backdrop-blur-xl cursor-pointer"
                  >
                    

                      {/* Image */}
                      <div className="w-full h-40 bg-slate-100 rounded-xl flex items-center justify-center overflow-hidden mb-3">
                        <img
                          src={selectedImage}
                          alt="product"
                          className="h-full object-contain"
                        />
                      </div>

                      {/* Title */}
                      <h3 className="text-sm font-semibold text-slate-800 line-clamp-2">
                        {item.name}
                      </h3>

                      {/* Price */}
                      <p className="text-green-600 font-semibold text-sm mt-1">
                        {item.price || "N/A"}
                      </p>

                      {/* Badge */}
                      <div className="mt-2 inline-block px-3 py-1 text-xs rounded-full 
                        bg-gradient-to-r from-blue-500 to-indigo-500 text-white">
                        {label}
                      </div>

                      <div className="mt-2 text-xs flex items-center justify-between">
                        <span className="text-slate-500">
                          Sentiment:
                        </span>
                        <span className="font-semibold text-emerald-600">
                          {item.sentiment_label}
                        </span>
                      </div>
                      <div className="mt-1 w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
                        <div
                          className="h-full bg-emerald-500"
                          style={{ width: `${item.sentiment_score * 100}%` }}
                        />
                      </div>

                      {/* Compact AI Insight */}
                      <div className="mt-3 text-xs text-slate-600 bg-blue-50 rounded-xl p-2">
                        <span className="font-semibold text-blue-500 block mb-1">
                          Why?
                        </span>

                        {item.explanation?.similar_items?.[0] && (
                          <p className="line-clamp-2">
                            Similar to: {item.explanation.similar_items[0]}
                          </p>
                        )}

                        <p className="mt-1">
                          Prediction score: ⭐ {item.explanation?.predicted_rating?.toFixed(2)}
                        </p>
                      </div>
                    </div>
                );
              })}
            </div>
            <p className="text-sm text-slate-500">
          Mode: Accuracy ({w1.toFixed(1)}), Diversity ({w2.toFixed(1)}), Novelty ({w3.toFixed(1)})
        </p>

        {/* Results */}
        {pref && (
          <div className="grid gap-8 animate-in fade-in zoom-in-95 duration-500">
            
            {/* Top Stat Row - Light Mode Apple Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
              
              <div className=" hover:scale-[1.02] transition-transform duration-200 bg-white/60 border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6 rounded-3xl backdrop-blur-2xl">
                <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">
                  Model Precision
                </p>
                <p className="text-2xl font-semibold mt-2 text-slate-800">
                  {pref?.precision?.toFixed(4) ?? "—"}
                </p>
              </div>

              <div className="hover:scale-[1.02] transition-transform duration-200 bg-white/60 border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6 rounded-3xl backdrop-blur-2xl">
                <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">
                  Recall
                </p>
                <p className="text-2xl font-semibold mt-2 text-blue-500">
                  {pref?.recall?.toFixed(4) ?? "—"}
                </p>
              </div>

              <div className="hover:scale-[1.02] transition-transform duration-200 bg-white/60 border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6 rounded-3xl backdrop-blur-2xl">
                <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">
                  F-Measure
                </p>
                <p className="text-2xl font-semibold mt-2 text-purple-500">
                  {pref?.f_measure?.toFixed(4) ?? "—"}
                </p>
              </div>

              <div className="hover:scale-[1.02] transition-transform duration-200 bg-white/60 border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6 rounded-3xl backdrop-blur-2xl">
                <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">
                  NDCG
                </p>
                <p className="text-2xl font-semibold mt-2 text-slate-700">
                  {pref?.ndcg?.toFixed(4) ?? "—"}
                </p>
              </div>

            </div>
            {selectedItem && (
                        <div
                            className="fixed inset-0 bg-black/40 flex items-center justify-center z-50"
                            onClick={() => setSelectedItem(null)}
                          >
                            <div
                              onClick={(e) => e.stopPropagation()}
                              className="bg-white rounded-2xl w-full max-w-lg shadow-xl relative animate-in fade-in zoom-in-95
                                        max-h-[90vh] flex flex-col"
                            >
                              {/* Header */}
                              <div className="p-5 border-b flex justify-between items-start">
                                <h2 className="text-lg font-bold pr-6">
                                  {selectedItem.name}
                                </h2>

                                <button
                                  onClick={() => setSelectedItem(null)}
                                  className="text-slate-500 hover:text-black"
                                >
                                  ✕
                                </button>
                              </div>

                              {/* Scrollable Content */}
                              <div className="overflow-y-auto p-5 space-y-4">
                                
                                {/* Image */}
                                <div className="h-48 bg-slate-100 rounded-xl flex items-center justify-center">
                                  <img
                                    src={
                                      selectedItem?.imageURLHighRes?.length
                                        ? selectedItem.imageURLHighRes[0]
                                        : "https://t3.ftcdn.net/jpg/05/04/28/96/360_F_504289605_zehJiK0tCuZLP2MdfFBpcJdOVxKLnXg1.jpg"
                                    }
                                    className="h-full object-contain"
                                  />
                                </div>

                                {/* Price */}
                                <p className="text-green-600 font-semibold">
                                  {selectedItem.price}
                                </p>

                                {/* AI Insight */}
                                <div className="text-sm text-slate-600">
                                  <p className="font-semibold text-blue-500 mb-1">✨ AI Insight</p>

                                  <div>
                                    <b>Similar to:</b>
                                    {selectedItem.explanation?.similar_items?.map((x, idx) => (
                                      <p key={idx}>- {x}</p>
                                    ))}
                                  </div>

                                  <div className="mt-2">
                                    <b>Because you liked:</b>
                                    {selectedItem.explanation?.because_you_liked?.map((x, idx) => (
                                      <p key={idx}>- {x}</p>
                                    ))}
                                  </div>

                                  <p className="mt-2">
                                    Predicted score: ⭐{" "}
                                    {selectedItem.explanation?.predicted_rating?.toFixed(2)}
                                  </p>
                                </div>

                                {/* SHAP */}
                                <div className="text-xs text-slate-500">
                                  <p className="font-semibold mb-1">📊 Why this score?</p>
                                  <p>Overall popularity: {selectedItem.shap_explanation?.components?.global_mean?.toFixed(2)}</p>
                                  <p>Your preference: {selectedItem.shap_explanation?.components?.user_bias?.toFixed(2)}</p>
                                  <p>Item popularity: {selectedItem.shap_explanation?.components?.item_bias?.toFixed(2)}</p>
                                  <p>Compatibility: {selectedItem.shap_explanation?.components?.interaction?.toFixed(2)}</p>
                                </div>

                                {/* Score Breakdown */}
                                <div className="bg-slate-50 rounded-xl p-3 text-xs text-slate-600">
                                  <p className="font-semibold text-slate-700 mb-2">⚖️ Final Score Breakdown</p>

                                  <p>⭐ Predicted: {selectedItem.score_breakdown?.predicted_rating?.toFixed(2)}</p>
                                  <p>📏 Normalized: {selectedItem.score_breakdown?.normalized_rating?.toFixed(2)}</p>
                                  <p>🌈 Diversity: {selectedItem.score_breakdown?.diversity?.toFixed(2)}</p>
                                  <p>🧭 Novelty: {selectedItem.score_breakdown?.novelty?.toFixed(2)}</p>
                                  <p>💬 Sentiment: {selectedItem.score_breakdown?.sentiment_norm?.toFixed(2)}</p>

                                  <div className="border-t my-2"></div>

                                  <p className="text-[11px]">
                                    Score = w₁·rating + w₂·diversity + w₃·novelty + α·sentiment
                                  </p>

                                  <p className="mt-2 text-sm font-semibold text-blue-600">
                                    Final: {selectedItem.score_breakdown?.final_score?.toFixed(3)}
                                  </p>
                                </div>

                              </div>
                            </div>
                          </div>
                      )}
                      
                  
          </div>
          
        )}
      </div>
      
    </div>
  );
}

export default App;