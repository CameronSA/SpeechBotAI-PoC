using Microsoft.Extensions.AI;
using OllamaSharp;
using System.Speech.Synthesis;
using NAudio.Wave;
using System.Text.Json;
using System.Net.Http.Headers;

IChatClient chatClient =
    new OllamaApiClient(new Uri("http://localhost:11434/"), "phi3:mini");

// Start the conversation with context for the AI model
List<ChatMessage> chatHistory = [];

while (true)
{
    var audioFile = "mic_input.wav";

    Console.WriteLine("🎤 Press Enter to start recording...");
    Console.ReadLine();
    using var waveIn = new WaveInEvent();
    waveIn.WaveFormat = new WaveFormat(16000, 1); // 16kHz mono
    using var writer = new WaveFileWriter(audioFile, waveIn.WaveFormat);
    waveIn.DataAvailable += (s, e) =>
    {
        writer.Write(e.Buffer, 0, e.BytesRecorded);
    };

    waveIn.StartRecording();

    Console.WriteLine("🎤 Recording... Press Enter to stop.");
    Console.ReadLine();
    waveIn.StopRecording();
    writer.Dispose();
    waveIn.Dispose();

    // Get user prompt and add to chat history
    var userPrompt = await SendToSTT(audioFile);
    Console.WriteLine($"Your prompt: {userPrompt}");
    if (string.IsNullOrWhiteSpace(userPrompt))
    {
        Console.WriteLine("No valid input detected. Please try again.");
        continue;
    }

    chatHistory.Add(new ChatMessage(ChatRole.User, userPrompt));

    // Stream the AI response and add to chat history
    Console.WriteLine("AI Response:");
    var response = "";
    await foreach (ChatResponseUpdate item in
        chatClient.GetStreamingResponseAsync(chatHistory))
    {
        Console.Write(item.Text);
        response += item.Text;
    }
    chatHistory.Add(new ChatMessage(ChatRole.Assistant, response));
    Console.WriteLine();

    // Convert response to speech
    using var speechSynthesizer = new SpeechSynthesizer();
    speechSynthesizer.Speak(response);
}

static async Task<string> SendToSTT(string filePath)
{
    using var client = new HttpClient();
    using var form = new MultipartFormDataContent();
    var audioData = await File.ReadAllBytesAsync(filePath);
    var audioContent = new ByteArrayContent(audioData);
    audioContent.Headers.ContentType = MediaTypeHeaderValue.Parse("audio/wav");
    form.Add(audioContent, "file", Path.GetFileName(filePath));

    var response = await client.PostAsync("http://localhost:8001/transcribe", form);
    response.EnsureSuccessStatusCode();
    var json = await response.Content.ReadAsStringAsync();

    var sttResult = JsonSerializer.Deserialize<STTResult>(json, new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true
    }) ?? new STTResult();
    return sttResult.Text;
}

public class STTResult
{
    public string Text { get; set; } = string.Empty;
}
