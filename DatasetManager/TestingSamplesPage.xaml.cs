// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for TestingSamplesPage.xaml
    /// </summary>
    public partial class TestingSamplesPage : Page
    {
        private readonly string _category;
        private List<SampleRecord> _samples;
        public List<SampleRecord> Samples => _samples.Where(s => s.Category == _category).ToList();

        private void Back_Click(object sender, RoutedEventArgs e)
        {
            NavigationService?.GoBack();
        }

        private void ListenButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is not Button button)
            {
                return;
            }

            if (button.DataContext is not SampleRecord sample)
            {
                return;
            }

            Process.Start(new ProcessStartInfo
            {
                FileName = sample.URI,
                UseShellExecute = true
            });
        }

        public TestingSamplesPage(List<SampleRecord> samples, string category)
        {
            InitializeComponent();
            _samples = samples;
            _category = category;
            DataContext = this;
        }
    }
}
