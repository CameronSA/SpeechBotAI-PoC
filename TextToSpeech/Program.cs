using System.Speech.Synthesis;

class Program
{
    static void Main()
    {
        var synth = new SpeechSynthesizer();
        var stringArr = new string[] { "Hello!", "This", "is", "your", "computer", "speaking.", "How", "can", "I", "assist", "you", "today?" };
        var sentenceArr = new string[] { "Hello! This is your computer speaking.", "How can I assist you today?" };
        synth.Speak("Hello! This is your computer speaking. How can I assist you today?");

        for (int i = 0; i < sentenceArr.Length; i++)
        {
            synth.Speak(sentenceArr[i]);
        }

        for (int i = 0; i < stringArr.Length; i++)
        {
            synth.Speak(stringArr[i]);
        }
    }
}
