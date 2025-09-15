import React from 'react';
import { Brain } from 'lucide-react';

const Header = () => {
  return (
    <header className="fixed top-0 w-full z-50 bg-gray-900/80 backdrop-blur-md border-b border-gray-800">
      <div className="container mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Brain className="w-8 h-8 text-purple-400" />
          <span className="text-xl font-bold text-white">FocusAgent</span>
        </div>
        <nav className="flex items-center space-x-8">
          <a href="#how-it-works" className="text-gray-300 hover:text-white transition-colors">How it Works</a>
          <a href="#testimonials" className="text-gray-300 hover:text-white transition-colors">Reviews</a>
        </nav>
      </div>
    </header>
  );
};

export default Header;