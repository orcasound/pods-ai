// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT
using CsvHelper;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Controls;

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

    /// <summary>
    /// Interaction logic for OverviewPage.xaml
    /// </summary>
    public partial class OverviewPage : Page
    {
        public ObservableCollection<StatsRow> Stats { get; } = new ObservableCollection<StatsRow>();

        private static List<SampleRecord> LoadSamples(string path)
        {
            using var reader = new StreamReader(path);
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);

            var records = csv.GetRecords<SampleRecord>();

            return records.ToList();
        }

        private void PopulateStats(
            List<SampleRecord> trainingSamples,
            List<SampleRecord> testingSamples,
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

        private void StatsGrid_SelectedCellsChanged(
            object sender,
            SelectedCellsChangedEventArgs e)
        {
            if (e.AddedCells.Count == 0)
            {
                return;
            }

            var cell = e.AddedCells[0];

            if (cell.Item is not StatsRow stat)
            {
                return;
            }

            var columnName = cell.Column.Header?.ToString();

            if (columnName == "# Training Samples")
            {
                // Navigate to the TrainingSamplesPage for the selected category.
                NavigationService?.Navigate(new TrainingSamplesPage(_trainingSamples, stat.Category));
            }
            else if (columnName == "# Testing Samples")
            {
                // Navigate to the TestingSamplesPage for the selected category.
                NavigationService?.Navigate(new TestingSamplesPage(_testingSamples, stat.Category));
            }
        }

        private List<SampleRecord> _trainingSamples;
        private List<SampleRecord> _testingSamples;
        private List<string> _trainingWavs;
        private List<string> _testingWavs;

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
                    .Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                int totalColumn = Array.IndexOf(headers, "total");
                for (int i = headerIndex + 2; i < lines.Length; i++)
                {
                    string line = lines[i];
                    if (string.IsNullOrWhiteSpace(line))
                    {
                        break;
                    }

                    string[] parts = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
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

        public OverviewPage()
        {
            InitializeComponent();

            string[] args = Environment.GetCommandLineArgs();
            string rootPath = @".";
            if (args.Length > 1)
            {
                rootPath = args[1];
            }

            try
            {
                _trainingSamples =
                    LoadSamples(Path.Combine(rootPath, "output", "csv", "training_3s_samples.csv"));
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Could not load output\\csv\\training_3s_samples.csv. Please ensure the CSV file exists and is accessible, and either run this application from the pods-ai directory, or specify the path to it on the command line.\n\nError: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                Environment.Exit(1);
            }

            try
            {
                _testingSamples =
                    LoadSamples(Path.Combine(rootPath, "output", "csv", "testing_60s_samples.csv"));
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Could not load testing samples. Please ensure the CSV file exists and is accessible, and either run this application from the pods-ai directory, or specify the path to it on the command line.\n\nError: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                Environment.Exit(1);
            }

            _trainingWavs = Directory.GetFiles(
                Path.Combine(rootPath, "output", "wav"),
                "*.wav",
                SearchOption.AllDirectories).ToList();

            _testingWavs = Directory.GetFiles(
                Path.Combine(rootPath, "output", "testing-wav"),
                "*.wav",
                SearchOption.AllDirectories).ToList();

            PopulateStats(_trainingSamples, _testingSamples, _trainingWavs, _testingWavs);

            ReadModelComparison();

            DataContext = this;
        }
    }
}
