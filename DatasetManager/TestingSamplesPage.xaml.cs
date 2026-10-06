// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

using System.Windows;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for TestingSamplesPage.xaml
    /// </summary>
    public partial class TestingSamplesPage : SamplesPageBase
    {
        protected override LogViewerControl LogViewer => LogViewerControl;
        protected override SamplesGridControl SamplesGrid => SamplesGridControl;
        protected override string WavFolderPath => @"output\testing-wav";
        public int FalsePositiveCount
        {
            get
            {
                StatsRow? row = Repository.Stats.FirstOrDefault(s => s.Category == Category);
                return (row != null) ? row.TestFalsePositiveCount : 0;
            }
        }

        public bool HasFalsePositives => FalsePositiveCount > 0;

        public TestingSamplesPage(DatasetRepository repository, string category)
            : base(repository, repository.TestingSamples, category)
        {
            InitializeComponent();
        }

        private string GetStartTimePSTStringForPastWeek()
        {
            TimeZoneInfo pacificZone = TimeZoneInfo.FindSystemTimeZoneById("Pacific Standard Time");
            DateTimeOffset oneWeekAgoPacific =
                TimeZoneInfo.ConvertTime(DateTimeOffset.UtcNow, pacificZone)
                .AddDays(-7);
            return oneWeekAgoPacific.ToString("yyyy_MM_dd_HH_mm_ss") + "_PST";
        }

        protected async void MoreFalsePositives_Click(object sender, RoutedEventArgs e)
        {
            if (Repository.ProposedTestingSamples.Count(sample => sample.Category == Category) == 0)
            {
                string timestamp = GetStartTimePSTStringForPastWeek();

                var result = await RunPythonAsync(@"src\process_false_positives.py", $"--set testing --start {timestamp} --end now --category {Category}");

                string? testingCsvSection = DatasetRepository.ExtractCsvSection(result, "Proposed rows for output/csv/testing_60s_samples.csv:");
                List<SampleRecord> newTestingSamples = (testingCsvSection != null)
                    ? DatasetRepository.LoadSamplesFromText(testingCsvSection)
                    : new();
                Repository.ProposeTestingSamples(newTestingSamples);
            }

            NavigationService?.Navigate(new AddTestingSamplesPage(Repository, Category));
        }

        protected async void MoreFalseNegatives_Click(object sender, RoutedEventArgs e)
        {
            if (Repository.ProposedTestingSamples.Count(sample => sample.Category == "resident") == 0)
            {
                string timestamp = GetStartTimePSTStringForPastWeek();

                var result = await RunPythonAsync(@"src\process_false_negatives.py", $"--start {timestamp} --end now --category {Category}");

                string? testingCsvSection = DatasetRepository.ExtractCsvSection(result, "Proposed rows for output/csv/testing_60s_samples.csv:");
                List<SampleRecord> newTestingSamples = (testingCsvSection != null)
                    ? DatasetRepository.LoadSamplesFromText(testingCsvSection)
                    : new();
                Repository.ProposeTestingSamples(newTestingSamples);
            }
            NavigationService?.Navigate(new AddTestingSamplesPage(Repository, Category));
        }

        protected async void FindMispredictions_Click(object sender, RoutedEventArgs e)
        {
            if (Repository.ProposedTrainingSamples.Count(sample => sample.Category == Category) == 0)
            {
                var result = await RunPythonAsync(@"src\process_testing_set_mispredictions.py", $"--category {Category}");

                string? trainingCsvSection = DatasetRepository.ExtractCsvSection(result, "Proposed rows for output/csv/training_3s_samples.csv:");
                List<SampleRecord> newTrainingSamples = (trainingCsvSection != null)
                    ? DatasetRepository.LoadSamplesFromText(trainingCsvSection)
                    : new();

                string? testingCsvSection = DatasetRepository.ExtractCsvSection(result, "Proposed rows to remove from output/csv/testing_60s_samples.csv:");
                List<SampleRecord> oldTestingSamples = (testingCsvSection != null)
                    ? DatasetRepository.LoadSamplesFromText(testingCsvSection)
                    : new();

                Repository.ProposeTrainingSamples(newTrainingSamples);
            }
            NavigationService?.Navigate(new AddTrainingSamplesPage(Repository, Category));
        }
    }
}
