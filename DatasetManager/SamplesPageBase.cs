// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;

namespace DatasetManager
{
    public abstract class SamplesPageBase : Page
    {
        protected readonly DatasetRepository Repository;
        protected abstract string WavFolderPath { get; }
        protected abstract LogViewerControl LogViewer { get; }
        protected abstract SamplesGridControl SamplesGrid { get; }
        public string Category { get; }

        protected void Back_Click(object sender, RoutedEventArgs e)
        {
            NavigationService?.GoBack();
        }


        protected async Task<string> RunPythonAsync(string scriptPath, string args)
        {
            try
            {
                SamplesGrid.InferenceEnabled = false;

                // Clear textbox.
                LogViewer.LogTextBox.Text = string.Empty;

                string[] commandLineArgs = Environment.GetCommandLineArgs();
                string rootPath = @".";
                if (commandLineArgs.Length > 1)
                {
                    rootPath = commandLineArgs[1];
                }
                string fullArguments = $"\"{scriptPath}\" {args}";
                LogViewer.AppendInfo($"> python {fullArguments.Trim()}");

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

            return LogViewer.LogTextBox.Text;
        }

        private string FindTagsInOutput(string prefix, string output, out double confidence)
        {
            confidence = 0;

            string? line = output
                .Split('\n')
                .FirstOrDefault(l => l.StartsWith(prefix));
            if (line != null)
            {
                string remainder = line.Substring(prefix.Length);

                int paren = remainder.IndexOf('(');
                if (paren < 0)
                {
                    confidence = 1.0;
                    return remainder.Trim();
                }

                string tags = remainder.Substring(0, paren);
                string confidencePrefix = "confidence:";
                int start = remainder.IndexOf(confidencePrefix);
                int end = remainder.IndexOf(')', start);
                if (start >= 0 && end > start)
                {
                    string value = remainder
                        .Substring(start + confidencePrefix.Length,
                                   end - start - confidencePrefix.Length)
                        .Trim();

                    double.TryParse(value, out confidence);
                }
                return tags.Trim();
            }
            return string.Empty;
        }

        protected async void SamplesGrid_InferenceRequested(object? sender, SampleRecord sample)
        {
            await RunPythonAsync("src\\run_inference.py", sample.GetWavFilePath(WavFolderPath));

            double confidence;
            string tag = FindTagsInOutput("Global prediction: ", LogViewer.LogTextBox.Text, out confidence);
            double dummyConfidence;
            string tags = FindTagsInOutput("Global predictions: ", LogViewer.LogTextBox.Text, out dummyConfidence);
            if (string.IsNullOrEmpty(tags))
            {
                tags = tag;
            }
            if (!string.IsNullOrEmpty(tags))
            {
                sample.Tags = tags;
                sample.Confidence = confidence * 100.0;
            }
        }

        public bool IsNotResident => !string.Equals(Category, "resident", StringComparison.OrdinalIgnoreCase);

        public ICollectionView Samples { get; }

        protected SamplesPageBase(DatasetRepository repository, ObservableCollection<SampleRecord> samples, string category)
        {
            Repository = repository;
            Category = category;

            Samples = CollectionViewSource.GetDefaultView(samples);
            Samples.Filter = o => o is SampleRecord sample && sample.Category == category;

            DataContext = this;
        }
    }
}
