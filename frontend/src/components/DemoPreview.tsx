import React, { useState, useEffect } from 'react';
import { Play, Volume2, Trophy, Zap, Target } from 'lucide-react';

const DemoPreview = () => {
  const [focusScore, setFocusScore] = useState(78);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setFocusScore(prev => {
        const newScore = prev + (Math.random() - 0.5) * 10;
        return Math.max(0, Math.min(100, newScore));
      });
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <section id="demo" className="py-24 bg-gradient-to-b from-gray-900 to-gray-800">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            See It In <span className="text-blue-400">Action</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Experience the magic of AI-powered focus training
          </p>
        </div>
        
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Demo Video/Preview */}
            <div className="relative">
              <div className="aspect-video bg-gray-800 rounded-2xl border border-gray-700 overflow-hidden relative">
                <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 to-blue-900/20"></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <button 
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="w-20 h-20 bg-white/10 backdrop-blur-sm border border-white/20 rounded-full flex items-center justify-center hover:bg-white/20 transition-all duration-300"
                  >
                    <Play className="w-8 h-8 text-white ml-1" />
                  </button>
                </div>
                
                {/* Simulated webcam overlay */}
                <div className="absolute top-4 left-4 w-32 h-24 bg-gray-700 rounded-lg border border-gray-600 overflow-hidden">
                  <div className="w-full h-full bg-gradient-to-br from-green-400/20 to-blue-400/20 flex items-center justify-center">
                    <div className="w-8 h-8 bg-green-400 rounded-full animate-pulse"></div>
                  </div>
                </div>
                
                {/* Focus indicator */}
                <div className="absolute top-4 right-4 bg-gray-900/80 backdrop-blur-sm rounded-lg p-3 border border-gray-600">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    <span className="text-sm text-white">Focused</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Live Dashboard */}
            <div className="space-y-6">
              {/* Live Score Ring */}
              <div className="bg-gray-800/50 border border-gray-700 rounded-2xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <Target className="w-5 h-5 text-purple-400 mr-2" />
                  Live Focus Score
                </h3>
                <div className="flex items-center justify-between">
                  <div className="relative w-24 h-24">
                    <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                      <circle
                        cx="50"
                        cy="50"
                        r="40"
                        stroke="currentColor"
                        strokeWidth="8"
                        fill="none"
                        className="text-gray-700"
                      />
                      <circle
                        cx="50"
                        cy="50"
                        r="40"
                        stroke="currentColor"
                        strokeWidth="8"
                        fill="none"
                        strokeDasharray={`${2 * Math.PI * 40}`}
                        strokeDashoffset={`${2 * Math.PI * 40 * (1 - focusScore / 100)}`}
                        className="text-purple-400 transition-all duration-1000"
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-2xl font-bold text-white">{Math.round(focusScore)}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-white">{Math.round(focusScore)}%</div>
                    <div className="text-sm text-gray-400">Current Focus</div>
                  </div>
                </div>
              </div>
              
              {/* Voice Quote */}
              <div className="bg-gray-800/50 border border-gray-700 rounded-2xl p-6">
                <div className="flex items-start space-x-3">
                  <Volume2 className="w-5 h-5 text-blue-400 mt-1" />
                  <div>
                    <p className="text-white italic mb-2">"You got this. Let's keep going."</p>
                    <p className="text-sm text-gray-400">AI Voice Coach</p>
                  </div>
                </div>
              </div>
              
              {/* Quick Stats */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <Trophy className="w-4 h-4 text-yellow-400" />
                    <span className="text-sm text-gray-400">Streak</span>
                  </div>
                  <div className="text-xl font-bold text-white">7 days</div>
                </div>
                <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <Zap className="w-4 h-4 text-purple-400" />
                    <span className="text-sm text-gray-400">Session</span>
                  </div>
                  <div className="text-xl font-bold text-white">45 min</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DemoPreview;