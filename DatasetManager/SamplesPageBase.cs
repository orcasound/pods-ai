// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;

namespace DatasetManager
{
    public abstract class SamplesPageBase : Page
    {
        protected abstract LogViewerControl LogViewer { get; }
        public string Category { get; }
        private List<SampleRecord> _samples;
        public List<SampleRecord> Samples => _samples.Where(s => s.Category == Category).ToList();

        protected void Back_Click(object sender, RoutedEventArgs e)
        {
            NavigationService?.GoBack();
        }

        private async Task RunPythonAsync(string scriptPath, string args)
        {
            string[] commandLineArgs = Environment.GetCommandLineArgs();
            string rootPath = @".";
            if (commandLineArgs.Length > 1)
            {
                rootPath = commandLineArgs[1];
            }

            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{scriptPath}\" {args}",
                WorkingDirectory = rootPath,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            var process = new Process
            {
                StartInfo = psi,
                EnableRaisingEvents = true
            };

            process.OutputDataReceived += (s, e) =>
            {
                if (e.Data != null)
                {
                    Dispatcher.Invoke(() =>
                    {
                        LogViewer.AppendInfo(e.Data);
                    });
                }
            };

            process.ErrorDataReceived += (s, e) =>
            {
                if (e.Data != null)
                {
                    Dispatcher.Invoke(() =>
                    {
                        LogViewer.AppendError(e.Data);
                    });
                }
            };

            process.Start();

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            await process.WaitForExitAsync();
        }

        protected async void SamplesGridControl_InferenceRequested(object? sender, SampleRecord sample)
        {
            await RunPythonAsync("src\\run_inference.py", sample.WavFilePath);
        }

        protected SamplesPageBase(List<SampleRecord> samples, string category)
        {
            _samples = samples;
            Category = category;
        }
    }
}
