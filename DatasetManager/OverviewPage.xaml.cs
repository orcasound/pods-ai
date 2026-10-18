// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

using System.Windows;
using System.Windows.Controls;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for OverviewPage.xaml
    /// </summary>
    public partial class OverviewPage : Page
    {
        public readonly DatasetRepository Repository;

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
                NavigationService?.Navigate(new TrainingSamplesPage(Repository, stat.Category));
            }
            else if (columnName == "# Testing Samples")
            {
                // Navigate to the TestingSamplesPage for the selected category.
                NavigationService?.Navigate(new TestingSamplesPage(Repository, stat.Category));
            }
        }

        public OverviewPage(DatasetRepository repository)
        {
            Repository = repository;
            InitializeComponent();
            DataContext = repository;
        }

        protected void Save_Click(object sender, RoutedEventArgs e)
        {
            Repository.Save();
        }
    }
}
