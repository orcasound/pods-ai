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
        public string Category { get; }
        private List<SampleRecord> _samples;
        public List<SampleRecord> Samples => _samples.Where(s => s.Category == Category).ToList();

        private void Back_Click(object sender, RoutedEventArgs e)
        {
            NavigationService?.GoBack();
        }

        public TestingSamplesPage(List<SampleRecord> samples, string category)
        {
            InitializeComponent();
            _samples = samples;
            Category = category;
            DataContext = this;
        }
    }
}
