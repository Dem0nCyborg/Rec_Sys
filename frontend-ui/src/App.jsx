import React, { useState } from 'react';

function App() {
  const [userId, setUserId] = useState();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  // A robust feature: Give the professor quick-test users
  const sampleUsers = ['A10JB7YPWZGRF4', 'A10ZJZNO4DAVB', 'A1E9D6RGEDFT2O'];
  const [selectedImages, setSelectedImages] = useState({});

  const fetchRecs = async (idToFetch) => {
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/recommend/${idToFetch}`);
      if (!res.ok) throw new Error("Backend not responding");
      const json = await res.json();
      setData(json);
    } catch (err) {
      alert("Error: Make sure your FastAPI backend is running!");
    } finally {
      setLoading(false);
    }
  };

  return (
    // Apple-style light background (#f5f5f7)
    <div className="min-h-screen bg-[#f5f5f7] text-slate-800 p-6 md:p-12 font-sans selection:bg-blue-200 overflow-hidden relative">
      
      {/* Soft Pastel Background Blobs for that Apple Vibrancy effect */}
      <div className="absolute top-[-5%] left-[-5%] w-96 h-96 bg-blue-300/40 rounded-full blur-[100px] pointer-events-none"></div>
      <div className="absolute bottom-[-10%] right-[-5%] w-96 h-96 bg-pink-300/30 rounded-full blur-[120px] pointer-events-none"></div>
      <div className="absolute top-[40%] left-[30%] w-80 h-80 bg-teal-200/40 rounded-full blur-[100px] pointer-events-none"></div>

      <div className="max-w-5xl mx-auto relative z-10">
        
        {/* Header & Search */}
        <header className="flex flex-col md:flex-row justify-between items-center mb-16 gap-6">
          <div>
            <h1 className="text-5xl md:text-6xl font-black tracking-tight text-slate-900">
              EchoRec <span className="text-blue-500">AI</span>
            </h1>
            <p className="text-slate-500 font-medium text-sm mt-2 uppercase tracking-[0.15em]">Neural Recommendation Engine</p>
          </div>
          
          <div className="flex flex-col items-end gap-3">
            {/* Apple-style search bar (White, heavily blurred, soft shadow) */}
            <div className="flex bg-white/60 p-2 rounded-2xl border border-white backdrop-blur-2xl shadow-[0_8px_30px_rgb(0,0,0,0.04)]">
              <input 
                type="text" 
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                className="bg-transparent px-4 py-2 focus:outline-none w-56 text-slate-700 font-medium placeholder-slate-400"
                placeholder="Enter User ID..."
              />
              <button 
                onClick={() => fetchRecs(userId)}
                disabled={loading}
                className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-6 rounded-xl transition-all active:scale-95 disabled:opacity-50 shadow-md"
              >
                {loading ? "Searching..." : "Generate"}
              </button>
            </div>
            
            {/* Quick Select feature for better UX */}
            <div className="flex gap-2 text-xs font-medium text-slate-400">
              <span>Try:</span>
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
        </header>

        {/* Results */}
        {data && (
          <div className="grid gap-8 animate-in fade-in zoom-in-95 duration-500">
            
            {/* Top Stat Row - Light Mode Apple Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
              
              <div className=" hover:scale-[1.02] transition-transform duration-200 bg-white/60 border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6 rounded-3xl backdrop-blur-2xl">
                <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">
                  Model Precision
                </p>
                <p className="text-2xl font-semibold mt-2 text-slate-800">
                  {data?.model_performance.precision.toFixed(4)}
                </p>
              </div>

              <div className="hover:scale-[1.02] transition-transform duration-200 bg-white/60 border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6 rounded-3xl backdrop-blur-2xl">
                <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">
                  Recall
                </p>
                <p className="text-2xl font-semibold mt-2 text-blue-500">
                  {data?.model_performance.recall.toFixed(4)}
                </p>
              </div>

              <div className="hover:scale-[1.02] transition-transform duration-200 bg-white/60 border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6 rounded-3xl backdrop-blur-2xl">
                <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">
                  F-Measure
                </p>
                <p className="text-2xl font-semibold mt-2 text-purple-500">
                  {data?.model_performance.f_measure.toFixed(4)}
                </p>
              </div>

              <div className="hover:scale-[1.02] transition-transform duration-200 bg-white/60 border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6 rounded-3xl backdrop-blur-2xl">
                <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">
                  NDCG
                </p>
                <p className="text-2xl font-semibold mt-2 text-slate-700">
                  {data?.model_performance.ndcg.toFixed(4)}
                </p>
              </div>

            </div>

            {/* Recommendations List - Rendering AI Reasoning for ALL items */}
            <div className="space-y-6 mt-4">
              {data.recommendations.map((item, i) => {

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
                    className="bg-white/60 border border-white p-6 rounded-[2rem] hover:bg-white/80 transition-all shadow-[0_8px_30px_rgb(0,0,0,0.04)] backdrop-blur-2xl"
                  >

                    {/* Top Section */}
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h3 className="text-xl font-bold text-slate-800 max-width-[700px]">
                          {item.name}
                        </h3>

                        <p className="text-slate-500 text-sm">
                          {item.item_id} • {item.brand || "Unknown Brand"}
                        </p>

                        <p className="text-green-600 font-semibold mt-1">
                          {item.price || "Price not available"}
                        </p>
                      </div>

                      {/* Score Badge */}
                      <div className="px-4 py-2 rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 text-white text-s font-semibold shadow">
                        {label}
                      </div>
                    </div>

                    {/* 🔥 Image Carousel */}
                    <div className="mt-4">

                      {/* 🔥 Main Large Image */}
                      <div className="w-full h-64 bg-white rounded-2xl flex items-center justify-center overflow-hidden border">
                        <img
                          src={selectedImage}
                          alt="main"
                          className="h-full object-contain transition-all duration-300"
                        />
                      </div>

                      {/* 🔽 Thumbnails */}
                      <div className="flex gap-3 mt-3 overflow-x-auto">
                        {images.map((img, idx) => (
                          <img
                            key={idx}
                            src={img}
                            alt="thumb"
                            onClick={() =>
                              setSelectedImages(prev => ({
                                ...prev,
                                [i]: img
                              }))
                            }
                            className={`w-16 h-16 object-cover rounded-lg border cursor-pointer transition-all
                              ${selectedImage === img
                                ? "border-blue-500 scale-105"
                                : "opacity-70 hover:opacity-100"
                              }`}
                          />
                        ))}
                      </div>

                    </div>
                    {/* AI Reason */}
                    <div className="mt-5 p-4 bg-blue-50/50 rounded-2xl border border-blue-100/50">
                      <p className="text-slate-600 text-sm leading-relaxed">
                        <span className="text-blue-500 font-bold uppercase text-xs block mb-1">
                          ✨ AI Insight
                        </span>
                        {item.ai_reason || "This product matches your preferences based on similar users and past interactions."}
                      </p>
                    </div>

                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;