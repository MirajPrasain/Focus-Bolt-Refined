from cv_project.langchain_utils import generate_focus_message

def test_focus_message():
    """Test the generate_focus_message function with sample parameters"""
    
    # Test parameters as specified
    duration = 25
    minute = 10
    vibe = "calm"
    cheat_count = 1
    
    print("🧪 Testing Focus Message Generation")
    print(f"Parameters: duration={duration}, minute={minute}, vibe='{vibe}', cheat_count={cheat_count}")
    print("-" * 50)
    
    try:
        # Generate the motivational message
        message = generate_focus_message(duration, vibe, minute, cheat_count)
        
        print("✅ Generated Message:")
        print(message)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error generating message: {e}")
        print("💡 Make sure Ollama is running with gemma:2b model installed")

if __name__ == "__main__":
    test_focus_message() 