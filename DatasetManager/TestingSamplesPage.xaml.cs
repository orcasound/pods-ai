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

        public bool HasFalsePositives
        {
            get
            {
                StatsRow? row = _repository.Stats.FirstOrDefault(s => s.Category == Category);
                return (row != null) && (row.TestFalsePositiveCount > 0);
            }
        }

        public TestingSamplesPage(DatasetRepository repository, string category)
            : base(repository, repository.TestingSamples, category)
        {
            InitializeComponent();
            DataContext = this;
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
            string timestamp = GetStartTimePSTStringForPastWeek();

            await RunPythonAsync(@"src\process_false_positives.py", $"--set testing --start {timestamp} --end now --category {Category}");
        }

        protected async void MoreFalseNegatives_Click(object sender, RoutedEventArgs e)
        {
            string timestamp = GetStartTimePSTStringForPastWeek();

            await RunPythonAsync(@"src\process_false_negatives.py", $"--set testing --start {timestamp} --end now --category {Category}");
        }

        protected async void FindMispredictions_Click(object sender, RoutedEventArgs e)
        {
            await RunPythonAsync(@"src\process_testing_set_mispredictions.py", $"--category {Category}");
        }
    }
}
