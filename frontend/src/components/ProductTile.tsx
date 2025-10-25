'use client';

import Image from 'next/image';

interface ProductCard {
  id: string;
  name: string;
  brand: string;
  price: number;
  rating?: number;
  image_url?: string;
  description?: string;
  category?: string;
  subcategory?: string;
}

interface ProductTileProps {
  product: ProductCard;
}

export default function ProductTile({ product }: ProductTileProps) {
  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(price);
  };

  const renderStars = (rating?: number) => {
    if (!rating) return null;
    
    const stars = Math.round(rating);
    return (
      <div className="flex items-center gap-1">
        {[...Array(5)].map((_, i) => (
          <span
            key={i}
            className={`text-sm ${
              i < stars ? 'text-yellow-400' : 'text-gray-300'
            }`}
          >
            ★
          </span>
        ))}
        <span className="text-xs text-gray-500 ml-1">({rating})</span>
      </div>
    );
  };


  const isValidImageUrl = (url: string) => {
    // Check if URL is from Amazon (which are placeholder URLs in our case)
    return !url.includes('m.media-amazon.com');
  };

  return (
    <div className="bg-white rounded-lg shadow-md hover:shadow-lg transition-all duration-300 overflow-hidden border border-gray-100 hover:border-blue-200 group">
      {/* Product Image */}
      <div className="relative h-48 bg-gray-50 overflow-hidden">
        {product.image_url && isValidImageUrl(product.image_url) ? (
          <Image
            src={product.image_url}
            alt={product.name}
            fill
            className="object-cover group-hover:scale-105 transition-transform duration-300"
            onLoad={(e) => {
              // Hide the fallback div when image loads successfully
              const target = e.target as HTMLImageElement;
              const fallbackDiv = target.nextElementSibling as HTMLElement;
              if (fallbackDiv) {
                fallbackDiv.style.display = 'none';
              }
            }}
            onError={(e) => {
              // Hide the image and show fallback div
              const target = e.target as HTMLImageElement;
              target.style.display = 'none';
              const fallbackDiv = target.nextElementSibling as HTMLElement;
              if (fallbackDiv) {
                fallbackDiv.style.display = 'flex';
              }
            }}
            unoptimized={true}
          />
        ) : null}
        
        {/* Always show fallback div, but hide it if image loads successfully */}
        <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-100 to-gray-200">
          <div className="text-center">
            <div className="text-4xl mb-2">📱</div>
            <div className="text-sm text-gray-500 font-medium">{product.brand}</div>
          </div>
        </div>
        
        {/* Category Badge */}
        {product.category && (
          <div className="absolute top-2 left-2">
            <span className="bg-blue-500 text-white text-xs px-2 py-1 rounded-full font-medium">
              {product.category}
            </span>
          </div>
        )}
      </div>

      {/* Product Info */}
      <div className="p-4">
        {/* Brand */}
        <div className="text-xs text-gray-500 uppercase tracking-wide font-medium mb-1">
          {product.brand}
        </div>

        {/* Product Name */}
        <h3 className="font-semibold text-gray-900 mb-2 line-clamp-2 group-hover:text-blue-600 transition-colors">
          {product.name}
        </h3>

        {/* Description */}
        {product.description && (
          <p className="text-sm text-gray-600 mb-3 line-clamp-2">
            {product.description}
          </p>
        )}

        {/* Rating */}
        {renderStars(product.rating)}

        {/* Price */}
        <div className="mt-3 flex items-center justify-between">
          <span className="text-lg font-bold text-green-600">
            {formatPrice(product.price)}
          </span>
          
          {/* Subcategory */}
          {product.subcategory && (
            <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded">
              {product.subcategory.replace('_', ' ')}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
