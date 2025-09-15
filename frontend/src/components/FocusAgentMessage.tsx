import React, { useEffect, useState } from 'react';

const AGENT_NAME_MAP = {
  calm: 'Chloe',
  beast: 'Carter',
  default: 'Nathaniel', //game 
};

const FocusAgentMessage = ({ message, vibe }) => {
  const agent = AGENT_NAME_MAP[vibe] || AGENT_NAME_MAP.default;

  if (!message) return null;

  return (
    <div className="mt-6 p-4 bg-gradient-to-br from-[#1c1c1e] to-[#2d2d30] text-white rounded-2xl shadow-md max-w-xl mx-auto animate-fade-in">
      <div className="flex items-start">
        <span className="text-2xl mr-3">🧠</span>
        <div>
          <div className="text-sm font-semibold text-[#9fa6b2] mb-1">{agent}</div>
          <div className="text-base leading-relaxed">{message}</div>
        </div>
      </div>
    </div>
  );
};


export default FocusAgentMessage 