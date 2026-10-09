// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for SamplesGridControl.xaml
    /// </summary>
    public partial class SamplesGridControl : UserControl
    {
        public event EventHandler<SampleRecord>? InferenceRequested;
        public event EventHandler<SampleRecord>? AcceptRequested;
        public event EventHandler<SampleRecord>? RejectRequested;

        public static readonly DependencyProperty InferenceEnabledProperty =
            DependencyProperty.Register(
                nameof(InferenceEnabled),
                typeof(bool),
                typeof(SamplesGridControl),
                new PropertyMetadata(true));

        public bool InferenceEnabled
        {
            get => (bool)GetValue(InferenceEnabledProperty);
            set => SetValue(InferenceEnabledProperty, value);
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

        private void InferButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button &&
                button.DataContext is SampleRecord sample)
            {
                InferenceRequested?.Invoke(this, sample);
            }
        }

        private void AcceptButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button &&
                button.DataContext is SampleRecord sample)
            {
                AcceptRequested?.Invoke(this, sample);
            }
        }

        private void RejectButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button &&
                button.DataContext is SampleRecord sample)
            {
                RejectRequested?.Invoke(this, sample);
            }
        }

        public static readonly DependencyProperty ShowReviewButtonsProperty =
            DependencyProperty.Register(
                nameof(ShowReviewButtons),
                typeof(bool),
                typeof(SamplesGridControl),
                new PropertyMetadata(false, OnShowReviewButtonsChanged));

        private static void OnShowReviewButtonsChanged(
        DependencyObject d,
        DependencyPropertyChangedEventArgs e)
        {
            ((SamplesGridControl)d).UpdateColumnVisibility();
        }

        public bool ShowReviewButtons
        {
            get => (bool)GetValue(ShowReviewButtonsProperty);
            set => SetValue(ShowReviewButtonsProperty, value);
        }

        private void UpdateColumnVisibility()
        {
            if (SamplesDataGrid.Columns.Count < 2)
            {
                return;
            }

            var visibility = ShowReviewButtons ? Visibility.Visible : Visibility.Collapsed;

            SamplesDataGrid.Columns[^1].Visibility = visibility; // Reject
            SamplesDataGrid.Columns[^2].Visibility = visibility; // Accept
        }

        public SamplesGridControl()
        {
            InitializeComponent();
            UpdateColumnVisibility();
        }
    }
}
