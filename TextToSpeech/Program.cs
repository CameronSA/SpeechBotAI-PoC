using System.Speech.Synthesis;

class Program
{
    static void Main()
    {
        var synth = new SpeechSynthesizer();
        synth.Speak("Hello! This is your computer speaking.");
    }
}
