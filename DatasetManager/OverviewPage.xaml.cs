using CsvHelper;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Controls;

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

        public OverviewPage()
        {
            InitializeComponent();


            _trainingSamples =
                LoadSamples(@"C:\Users\dthal\git\orcasound\pods-ai\output\csv\training_3s_samples.csv");

            _testingSamples =
                LoadSamples(@"C:\Users\dthal\git\orcasound\pods-ai\output\csv\testing_60s_samples.csv");

            PopulateStats(_trainingSamples, _testingSamples);

            DataContext = this;
        }
    }
}
