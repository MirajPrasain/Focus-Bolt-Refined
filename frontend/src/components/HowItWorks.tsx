import React from 'react';
import { Camera, Volume2, Target, Brain } from 'lucide-react';

const features = [
  {
    icon: Camera,
    title: "Real-time Face & Eye Tracking",
    description: "Advanced Google MediaPipe technology tracks your focus patterns without being intrusive. Your privacy stays yours."
  },
  {
    icon: Volume2,
    title: "ElevenLabs Voice Coaching",
    description: "Natural, encouraging voice reminders powered by ElevenLabs when you drift. Like having a supportive coach by your side."
  },
  {
    icon: Target,
    title: "Smart Dashboard Insights",
    description: "Gamified progress tracking with scores, badges, and streaks. See your focus skills level up over time."
  },
  {
    icon: Brain,
    title: "AI Session Summaries",
    description: "Intelligent post-session analysis with personalized tips to improve your next deep work session."
  }
];

const HowItWorks = () => {
  return (
    <section id="how-it-works" className="py-24 bg-gray-900">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            How <span className="text-purple-400">FocusAgent</span> Works
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Four powerful features working together to create your perfect focus environment
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <div key={index} className="group">
              <div className="bg-gray-800/50 border border-gray-700 rounded-2xl p-8 h-full transition-all duration-300 hover:bg-gray-800/70 hover:border-purple-500/50 hover:transform hover:scale-105">
                <div className="w-12 h-12 bg-purple-600/20 rounded-xl flex items-center justify-center mb-6 group-hover:bg-purple-600/30 transition-colors">
                  <feature.icon className="w-6 h-6 text-purple-400" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-4">{feature.title}</h3>
                <p className="text-gray-400 leading-relaxed">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;