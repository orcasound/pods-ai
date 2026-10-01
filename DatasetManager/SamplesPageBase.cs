// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Shapes;

namespace DatasetManager
{
    public abstract class SamplesPageBase : Page
    {
        protected abstract string WavFolderPath { get; }
        protected abstract LogViewerControl LogViewer { get; }
        protected abstract SamplesGridControl SamplesGrid { get; }
        public string Category { get; }
        private readonly List<SampleRecord> _filteredSamples;
        public List<SampleRecord> Samples => _filteredSamples;

        protected void Back_Click(object sender, RoutedEventArgs e)
        {
            NavigationService?.GoBack();
        }

        private async Task RunPythonAsync(string scriptPath, string args)
        {
            try
            {
                SamplesGrid.InferenceEnabled = false;

                string[] commandLineArgs = Environment.GetCommandLineArgs();
                string rootPath = @".";
                if (commandLineArgs.Length > 1)
                {
                    rootPath = commandLineArgs[1];
                }
                string fullArguments = $"\"{scriptPath}\" {args}";
                LogViewer.AppendInfo($"python {fullArguments}");

                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = fullArguments,
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
            finally
            {
                SamplesGrid.InferenceEnabled = true;
            }
        }

        private string FindTagsInOutput(string prefix, string output)
        {
            string? line = output
                .Split('\n')
                .FirstOrDefault(l => l.StartsWith(prefix));
            if (line != null)
            {
                string remainder = line.Substring(prefix.Length);

                int paren = remainder.IndexOf('(');
                if (paren < 0)
                {
                    return remainder.Trim();
                }

                string tags = remainder.Substring(0, paren);
                return tags.Trim();
            }
            return string.Empty;
        }

        protected async void SamplesGrid_InferenceRequested(object? sender, SampleRecord sample)
        {
            await RunPythonAsync("src\\run_inference.py", sample.GetWavFilePath(WavFolderPath));

            string tags = FindTagsInOutput("Global predictions: ", LogViewer.LogTextBox.Text);
            if (string.IsNullOrEmpty(tags))
            {
                tags = FindTagsInOutput("Global prediction: ", LogViewer.LogTextBox.Text);
            }
            if (!string.IsNullOrEmpty(tags))
            {
                sample.Tags = tags;
            }
        }

        protected SamplesPageBase(List<SampleRecord> samples, string category)
        {
            Category = category;
            _filteredSamples = samples.Where(s => s.Category == category).ToList();
        }
    }
}
