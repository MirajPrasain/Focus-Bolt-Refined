import React from 'react';
import { Star, Quote } from 'lucide-react';

const testimonials = [
  {
    quote: "It feels like a personal trainer for my brain. I've never been this consistent with deep work.",
    author: "Sarah Chen",
    role: "Software Engineer",
    rating: 5
  },
  {
    quote: "I've never studied this consistently before. The AI coaching feels so supportive, not judgmental.",
    author: "Marcus Rodriguez",
    role: "Graduate Student",
    rating: 5
  },
  {
    quote: "The gamification aspect is genius. I actually look forward to my focus sessions now.",
    author: "Emily Thompson",
    role: "Product Designer",
    rating: 5
  }
];

const Testimonials = () => {
  return (
    <section id="testimonials" className="py-24 bg-gray-800">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Loved by <span className="text-green-400">Focused</span> People
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            See how FocusAgent is transforming the way people approach deep work
          </p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {testimonials.map((testimonial, index) => (
            <div key={index} className="group">
              <div className="bg-gray-900/50 border border-gray-700 rounded-2xl p-8 h-full transition-all duration-300 hover:bg-gray-900/70 hover:border-green-500/50 hover:transform hover:scale-105">
                <div className="mb-4">
                  <Quote className="w-8 h-8 text-green-400 mb-4" />
                  <div className="flex space-x-1 mb-4">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Star key={i} className="w-4 h-4 text-yellow-400 fill-current" />
                    ))}
                  </div>
                </div>
                
                <p className="text-gray-300 text-lg leading-relaxed mb-6 italic">
                  "{testimonial.quote}"
                </p>
                
                <div className="border-t border-gray-700 pt-4">
                  <p className="text-white font-semibold">{testimonial.author}</p>
                  <p className="text-gray-400 text-sm">{testimonial.role}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Testimonials;