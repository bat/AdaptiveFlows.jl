# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

@recipe function plot_flow(flow::AbstractFlow, samples::AbstractMatrix;
    d_sel = nothing, 
    n_bins_1D = 100, 
    n_bins_2D = 100, 
    colorbar = false,
    h_x = true,
    h_y = true,
    fs = nothing,
    size_plot = (1000, 1000),
    p = x -> 1/sqrt(2pi) * exp(-x^2/2),
    training_metadata = nothing
)
    n_dims = size(samples,1)

    if isnothing(d_sel)
        d_sel = 1:Integer(minimum([6, n_dims]))
    end
    n_plots = length(d_sel)

    if isnothing(fs)
        fs = n_plots > 5 ? 6 : n_plots > 3 ? 7 : 11
    end

    samples_transformed = flow(samples)

    stds_in = vec(std(samples, dims = 2))
    means_in = vec(mean(samples, dims = 2))
    samples_in = InvMulAdd(Diagonal(stds_in), means_in)(samples)

    stds_out = vec(std(samples_transformed, dims = 2))   
    means_out = vec(mean(samples_transformed, dims = 2))
    samples_out = InvMulAdd(Diagonal(stds_out), means_out)(samples_transformed)

    if !isnothing(training_metadata)
        training_metadata_labels = ["[Training metadata] ", "Loss: $(training_metadata[:loss]) ", "Optimizer: ", "$(training_metadata[:optimizer]) ", "# Batches: $(training_metadata[:nbatches]) ", "# Epochs: $(training_metadata[:nepochs]) ", "Sequential?: $(training_metadata[:sequential]) ", "Shuffle samples?: $(training_metadata[:shuffle_samples]) "]
    end
    samples_metadata = ["[Samples metadata]         ", "# Samples: $(size(samples,2)) ", "# Dimensions: $n_dims ", "Displayed Dimensions: ", "$(d_sel) ", " ", " ", " "]

    layout --> (n_plots + 1, n_plots)
    size --> size_plot

    for i in 1:n_plots
        bin_range_1D = range(minimum([minimum(samples_in[i,:]), minimum(samples_out[i,:])]), stop = maximum([maximum(samples_in[i,:]), maximum(samples_out[i,:])]), length = n_bins_1D)

        # Diagonal
        subplot := i + (i-1) * n_plots
        tickfontsize --> fs
        labelfontsize --> fs
        if h_x
            @series begin
                seriestype --> :stephist
                bins --> bin_range_1D
                normalize --> :pdf
                label --> false
                color --> :blue
                fill --> true
                alpha --> 0.3
                samples_in[d_sel[i],:]
            end
        end
        if h_y
            @series begin
                seriestype --> :stephist
                bins --> bin_range_1D
                normalize --> :pdf
                label --> false
                color --> :red
                fill --> true
                alpha --> 0.3
                samples_out[d_sel[i],:]
            end
        end
        @series begin
            seriestype --> :line
            lw --> 1.5
            color --> :black
            label --> false
            p
        end

        for j in i+1:n_plots
            # Input, lower off-diagonal
            subplot := i + (j - 1) * n_plots
            @series begin
                seriestype --> :histogram2d
                bins --> n_bins_2D
                color --> :blues
                colorbar --> colorbar
                background --> :white
                aspect_ratio --> :equal
                tickfontsize --> fs
                labelfontsize --> fs
                xlabel --> "x$((d_sel[i]))"
                ylabel --> "x$((d_sel[j]))"
                (samples_in[d_sel[i],:], samples_in[d_sel[j],:])
            end

            # Output upper off-diagonal
            subplot := j + (i - 1) * n_plots
            
            @series begin
                seriestype --> :histogram2d
                bins --> n_bins_2D
                color --> :reds
                colorbar --> colorbar
                aspect_ratio --> :equal
                tickfontsize --> fs
                labelfontsize --> fs
                xlabel -->  "y$(d_sel[i])"
                ylabel --> "y$(d_sel[j])"
                (samples_out[d_sel[i],:], samples_out[d_sel[j],:])
            end
        end

        subplot := n_plots^2 + n_plots + 1 - i

        if i == n_plots && !isnothing(training_metadata)
            for k in 1:8
                @series begin
                    seriestype --> :scatter
                    ticks --> false
                    framestyle --> :none 
                    legend --> :bottomleft
                    legendfontsize --> fs
                    label --> training_metadata_labels[k]
                    markeralpha --> 0
                    [0]
                end
            end
        elseif i == 1
            for k in 1:8
                @series begin
                    seriestype --> :scatter
                    ticks --> false
                    framestyle --> :none 
                    legend --> :bottomright
                    legendfontsize --> fs
                    label --> samples_metadata[k]
                    markeralpha --> 0
                    [0]
                end
            end
        else
            @series begin
                seriestype --> :scatter
                ticks --> false
                framestyle --> :none 
                legend --> false
                markeralpha --> 0
                [0]
            end
        end
    end
end
                                    
@recipe function plot_flow_res(res::Union{NamedTuple{(:result, :optimizer_state, :loss_hist, :training_metadata), Tuple{CompositeFlow, Vector{Any}, Vector{Any}, Dict{Symbol, Any}}}, 
                                          NamedTuple{(:result, :optimizer_state, :loss_hist, :training_metadata), Tuple{CompositeFlow, NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple}}}}}, Vector{Float64}, Dict{Symbol, Any}}}}, 
    samples::AbstractMatrix;
    d_sel = nothing,
    n_bins_1D = 100, 
    n_bins_2D = 100, 
    colorbar = false,
    h_x = true,
    h_y = true,
    fs = nothing,
    size_plot = (1000, 1000),
    p = x -> 1/sqrt(2pi) * exp(-x^2/2)
    )
    @series begin
        n_bins_1d := n_bins_1D
        n_bins_2d := n_bins_2D
        d_sel := d_sel
        colorbar := colorbar
        h_x := h_x
        h_y := h_y
        fs := fs
        size := size_plot
        p := p
        training_metadata := res.training_metadata
        (res.result, samples)
    end
end
