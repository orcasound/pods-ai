// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

using CsvHelper;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Windows;

namespace DatasetManager
{
    public class StatsRow
    {
        public string Category { get; set; } = "";
        public int TrainingSampleCount { get; set; }
        public int TestingSampleCount { get; set; }
        public int TrainingWavCount { get; set; }
        public int TestingWavCount { get; set; }
        public int TestCorrectCount { get; set; }
        public int TestTotalCount { get; set; }
        public int TestFalsePositiveCount { get; set; }
        public int TestFalseNegativeCount { get; set; }

        public double TestCorrectRatio => 1.0 * TestCorrectCount / TestTotalCount;
        public double TestFalsePositiveRatio => 1.0 * TestFalsePositiveCount / TestTotalCount;
        public double TestFalseNegativeRatio => 1.0 * TestFalseNegativeCount / TestTotalCount;
    }

    public class DatasetRepository
    {
        public ObservableCollection<SampleRecord> TrainingSamples = new();
        public ObservableCollection<SampleRecord> TestingSamples = new();
        public ObservableCollection<SampleRecord> ProposedTrainingSamples = new();
        public ObservableCollection<SampleRecord> ProposedTestingSamples = new();
        public ObservableCollection<SampleRecord> RejectedTrainingSamples = new();
        public ObservableCollection<SampleRecord> RejectedTestingSamples = new();
        private List<string> _trainingWavs = new();
        private List<string> _testingWavs = new();
        public ObservableCollection<StatsRow> Stats { get; } = new ObservableCollection<StatsRow>();

        private static List<SampleRecord> LoadSamplesFromCsv(CsvReader csv)
        {
            var records = csv.GetRecords<SampleRecord>();
            return records.ToList();
        }
        public static List<SampleRecord> LoadSamplesFromText(string csvText)
        {
            using var reader = new StringReader(csvText);
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
            return LoadSamplesFromCsv(csv);
        }
        private static List<SampleRecord> LoadSamples(string path)
        {
            using var reader = new StreamReader(path);
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
            return LoadSamplesFromCsv(csv);
        }
        public static string? ExtractCsvSection(string text, string header)
        {
            string marker = header + Environment.NewLine;

            int start = text.IndexOf(marker, StringComparison.Ordinal);
            if (start < 0)
            {
                return null;
            }

            start += marker.Length;

            int end = text.IndexOf(
                Environment.NewLine + Environment.NewLine,
                start,
                StringComparison.Ordinal);
            if (end < 0)
            {
                end = text.Length;
            }

            return text[start..end].Trim();
        }

        private void ReadModelComparison()
        {
            string[] args = Environment.GetCommandLineArgs();
            string rootPath = @".";
            if (args.Length > 1)
            {
                rootPath = args[1];
            }
            string modelComparisonPath = Path.Combine(rootPath, "model-comparison.txt");
            if (File.Exists(modelComparisonPath))
            {
                string text = File.ReadAllText(modelComparisonPath);
                string[] lines = text.Split('\n');

                // Find the line "Confusion Matrix for podsai (rows=actual, cols=predicted):"
                int headerIndex = Array.FindIndex(lines, l => l.StartsWith("Confusion Matrix for podsai"));
                if (headerIndex < 0 || headerIndex + 1 >= lines.Length)
                {
                    return;
                }

                // Parse headers so we can the labels and offsets for each column.
                string[] headers = lines[headerIndex + 1]
                    .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                int totalColumn = Array.IndexOf(headers, "total");
                for (int i = headerIndex + 2; i < lines.Length; i++)
                {
                    string line = lines[i];
                    if (string.IsNullOrWhiteSpace(line))
                    {
                        break;
                    }

                    string[] parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length < 2)
                    {
                        continue;
                    }
                    string category = parts[0];
                    int column = Array.IndexOf(headers, category);
                    int correct = column >= 0 ? int.Parse(parts[column + 1]) : 0;
                    int falsePositives = 0;
                    int falseNegatives = 0;
                    int total = int.Parse(parts[^1]); // last value
                    if (category == "resident" || category == "transient" || category == "humpback")
                    {
                        falseNegatives = total - correct;
                    }
                    else
                    {
                        int residentColumn = Array.IndexOf(headers, "resident");
                        int residents = (category != "resident" && residentColumn >= 0) ? int.Parse(parts[1 + residentColumn]) : 0;
                        int transientColumn = Array.IndexOf(headers, "transient");
                        int transients = (category != "transient" && transientColumn >= 0) ? int.Parse(parts[1 + transientColumn]) : 0;
                        int humpbackColumn = Array.IndexOf(headers, "humpback");
                        int humpbacks = (category != "humpback" && humpbackColumn >= 0) ? int.Parse(parts[1 + humpbackColumn]) : 0;
                        falsePositives = residents + transients + humpbacks;
                    }

                    StatsRow? row = Stats.FirstOrDefault(s => s.Category == category);
                    if (row != null)
                    {
                        row.TestCorrectCount = correct;
                        row.TestTotalCount = total;
                        row.TestFalsePositiveCount = falsePositives;
                        row.TestFalseNegativeCount = falseNegatives;
                    }
                    else
                    {
                        Stats.Add(new StatsRow
                        {
                            Category = category,
                            TestCorrectCount = correct,
                            TestTotalCount = total,
                            TestFalsePositiveCount = falsePositives,
                            TestFalseNegativeCount = falseNegatives,
                        });
                    }
                }
            }
        }

        private void PopulateStats(
            ObservableCollection<SampleRecord> trainingSamples,
            ObservableCollection<SampleRecord> testingSamples,
            List<string> trainingWavs,
            List<string> testingWavs)
        {
            var categories =
                trainingSamples.Select(s => s.Category)
                .Union(testingSamples.Select(s => s.Category))
                .OrderBy(c => c);

            foreach (var category in categories)
            {

                Stats.Add(new StatsRow
                {
                    Category = category,
                    TrainingSampleCount =
                        trainingSamples.Count(s => s.Category == category),

                    TestingSampleCount =
                        testingSamples.Count(s => s.Category == category),

                    TrainingWavCount =
                        trainingWavs.Count(s => s.Contains(category)),

                    TestingWavCount =
                        testingWavs.Count(s => s.Contains(category)),
                });
            }
        }

        public static DatasetRepository Load()
        {
            var repository = new DatasetRepository();

            string[] args = Environment.GetCommandLineArgs();
            string rootPath = @".";
            if (args.Length > 1)
            {
                rootPath = args[1];
            }

            try
            {
                List<SampleRecord> samples = LoadSamples(Path.Combine(rootPath, "output", "csv", "training_3s_samples.csv"));
                repository.TrainingSamples = new ObservableCollection<SampleRecord>(samples);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Could not load output\\csv\\training_3s_samples.csv. Please ensure the CSV file exists and is accessible, and either run this application from the pods-ai directory, or specify the path to it on the command line.\n\nError: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                Environment.Exit(1);
            }

            try
            {
                List<SampleRecord> samples =
                    LoadSamples(Path.Combine(rootPath, "output", "csv", "testing_60s_samples.csv"));
                repository.TestingSamples = new ObservableCollection<SampleRecord>(samples);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Could not load testing samples. Please ensure the CSV file exists and is accessible, and either run this application from the pods-ai directory, or specify the path to it on the command line.\n\nError: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                Environment.Exit(1);
            }

            repository._trainingWavs = Directory.GetFiles(
                Path.Combine(rootPath, "output", "wav"),
                "*.wav",
                SearchOption.AllDirectories).ToList();

            repository._testingWavs = Directory.GetFiles(
                Path.Combine(rootPath, "output", "testing-wav"),
                "*.wav",
                SearchOption.AllDirectories).ToList();

            repository.PopulateStats(repository.TrainingSamples, repository.TestingSamples, repository._trainingWavs, repository._testingWavs);

            repository.ReadModelComparison();

            return repository;
        }

        public void ProposeTrainingSamples(List<SampleRecord> newTrainingSamples)
        {
            foreach (var sample in newTrainingSamples)
            {
                ProposedTrainingSamples.Add(sample);
            }
        }

        public void ProposeTestingSamples(List<SampleRecord> newTestingSamples)
        {
            foreach (var sample in newTestingSamples)
            {
                ProposedTestingSamples.Add(sample);
            }
        }

        public void AcceptTestingSample(SampleRecord testingSample)
        {
            TestingSamples.Add(testingSample);
            ProposedTestingSamples.Remove(testingSample);
        }

        public void RejectTestingSample(SampleRecord testingSample)
        {
            RejectedTestingSamples.Add(testingSample);
            ProposedTestingSamples.Remove(testingSample);
        }

        public void AcceptTrainingSample(SampleRecord trainingSample)
        {
            TrainingSamples.Add(trainingSample);
            ProposedTrainingSamples.Remove(trainingSample);

            // TODO: remove any overlap from testing samples
        }

        public void RejectTrainingSample(SampleRecord trainingSample)
        {
            RejectedTrainingSamples.Add(trainingSample);
            ProposedTrainingSamples.Remove(trainingSample);
        }
    }
}
