// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for AddTestingSamplesPage.xaml
    /// </summary>
    public partial class AddTestingSamplesPage : SamplesPageBase
    {
        protected override LogViewerControl LogViewer => new LogViewerControl();
        protected override SamplesGridControl SamplesGrid => SamplesGridControl;
        protected override string WavFolderPath => @"output\testing-wav";

        private void SamplesGrid_AcceptRequested(object? sender, SampleRecord sample)
        {
            Repository.AcceptTestingSample(sample);
        }

        private void SamplesGrid_RejectRequested(object? sender, SampleRecord sample)
        {
            Repository.RejectTestingSample(sample);
        }

        public AddTestingSamplesPage(DatasetRepository repository, string category)
            : base(repository, repository.ProposedTestingSamples, category)
        {
            InitializeComponent();
        }
    }
}
