using CsvHelper;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Windows;

namespace DatasetManager
{
    public class SampleRecord
    {
        public string Category { get; set; } = "";
        public string NodeName { get; set; } = "";
        public string StartTimestamp { get; set; } = "";
        public string URI { get; set; } = "";
        public string Description { get; set; } = "";
        public string Notes { get; set; } = "";
        public double? Confidence { get; set; }
    }
    public class StatsRow
    {
        public string Category { get; set; } = "";
        public int TrainingSampleCount { get; set; }
        public int TestingSampleCount { get; set; }
    }

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
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
            List<SampleRecord> testingSamples)
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
                        testingSamples.Count(s => s.Category == category)
                });
            }
        }

        public MainWindow()
        {
            InitializeComponent();
            
            var trainingSamples =
                LoadSamples(@"C:\Users\dthal\git\orcasound\pods-ai\output\csv\training_3s_samples.csv");

            var testingSamples =
                LoadSamples(@"C:\Users\dthal\git\orcasound\pods-ai\output\csv\testing_60s_samples.csv");

            PopulateStats(trainingSamples, testingSamples);

            DataContext = this;
        }
    }
}