import React, { useState } from 'react';

function App() {
  const [userId, setUserId] = useState('A281N877SNCA7H');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  // A robust feature: Give the professor quick-test users
  const sampleUsers = ['A281N877SNCA7H', 'B392X110POQL5', 'C884M992WEAZ1'];

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
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white/60 border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6 rounded-3xl backdrop-blur-2xl">
                <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">Model Precision</p>
                <p className="text-3xl font-semibold mt-2 text-slate-800">98.4%</p>
              </div>
              <div className="bg-white/60 border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6 rounded-3xl backdrop-blur-2xl">
                <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">MAE Score</p>
                <p className="text-3xl font-semibold mt-2 text-blue-500">0.0939</p>
              </div>
              <div className="bg-white/60 border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-6 rounded-3xl backdrop-blur-2xl">
                <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">Inference Engine</p>
                <p className="text-3xl font-semibold mt-2 text-purple-500">Gemma 3</p>
              </div>
            </div>

            {/* Recommendations List - Rendering AI Reasoning for ALL items */}
            <div className="space-y-6 mt-4">
              {data.recommendations.map((item, i) => (
                <div key={i} className="bg-white/60 border border-white p-8 rounded-[2rem] hover:bg-white/80 transition-all shadow-[0_8px_30px_rgb(0,0,0,0.04)] backdrop-blur-2xl">
                  
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="text-2xl font-bold text-slate-800 mb-1">{item.name}</h3>
                      <p className="text-slate-500 font-mono text-sm">{item.id} • {item.price}</p>
                    </div>
                    <div className="h-14 w-14 rounded-full border border-blue-200 flex items-center justify-center bg-blue-50 text-blue-600 font-bold text-lg">
                      {item.confidence}
                    </div>
                  </div>

                  {/* Gemma Reasoning for EVERY item */}
                  <div className="mt-6 p-5 bg-blue-50/50 rounded-2xl border border-blue-100/50 relative">
                    <p className="text-slate-600 leading-relaxed text-md">
                      <span className="text-blue-500 font-bold uppercase tracking-wider text-xs block mb-1">✨ Gemma 3 Reasoning</span> 
                      "{item.ai_reason}"
                    </p>
                  </div>

                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;